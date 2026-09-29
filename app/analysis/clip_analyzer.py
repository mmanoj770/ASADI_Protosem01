import json
import re
from typing import List
from app.models.schemas import (
    TranscriptSegment,
    ClipCandidate,
    ClipAnalysisResponse,
    timestamp_to_seconds,
    seconds_to_timestamp
)
from app.analysis.prompts import SYSTEM_PROMPT, build_clip_analysis_prompt
from app.analysis.ollama_service import generate_ollama_completion, check_ollama_status

def clean_json_response(raw_text: str) -> str:
    """Extract clean JSON string from LLM output, removing markdown code blocks."""
    text = raw_text.strip()
    # Remove ```json ... ``` wrapper
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Find first { or [ and last } or ]
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        return match.group(1)
    return text

def parse_llm_clip_response(raw_response: str) -> List[ClipCandidate]:
    """Parse and validate JSON response into list of ClipCandidate objects."""
    cleaned = clean_json_response(raw_response)
    data = json.loads(cleaned)
    
    if isinstance(data, dict) and "clips" in data:
        clip_list = data["clips"]
    elif isinstance(data, list):
        clip_list = data
    else:
        clip_list = []

    results = []
    for item in clip_list:
        if not isinstance(item, dict):
            continue
        start_ts = str(item.get("start_time", "00:00:00"))
        end_ts = str(item.get("end_time", "00:00:30"))
        start_sec = timestamp_to_seconds(start_ts)
        end_sec = timestamp_to_seconds(end_ts)
        duration = round(max(0.0, end_sec - start_sec), 1)

        raw_score = min(100, max(0, int(item.get("score", 80))))
        base_s = round(raw_score / 10.0, 1)
        
        # Extract breakdown or construct from metrics
        breakdown = item.get("breakdown")
        if not isinstance(breakdown, dict):
            breakdown = {
                "hook": round(max(1.0, min(10.0, float(item.get("hook_score", base_s + 0.3 if "?" in str(item.get("hook", "")) else base_s)))), 1),
                "engagement": round(base_s, 1),
                "info_value": round(max(1.0, min(10.0, base_s - 0.2 if base_s > 5 else base_s)), 1),
                "emotion": round(max(1.0, min(10.0, base_s + 0.1 if base_s < 9 else base_s)), 1),
                "clarity": round(max(1.0, min(10.0, base_s + 0.3)), 1),
                "standalone": round(max(1.0, min(10.0, base_s - 1.0 if item.get("context_required") else base_s + 0.2)), 1)
            }
        else:
            # Ensure values are floats 0-10
            breakdown = {k: float(v) if float(v) <= 10 else round(float(v)/10.0, 1) for k, v in breakdown.items()}

        candidate = ClipCandidate(
            title=str(item.get("title", "Untitled Clip")),
            start_time=seconds_to_timestamp(start_sec),
            end_time=seconds_to_timestamp(end_sec),
            start_sec=start_sec,
            end_sec=end_sec,
            duration=duration,
            score=raw_score,
            topic=str(item.get("topic", "General")),
            hook=str(item.get("hook", "")),
            reason=str(item.get("reason", "")),
            context_required=bool(item.get("context_required", False)),
            breakdown=breakdown
        )
        results.append(candidate)
    return results

def fallback_heuristic_analyzer(
    segments: List[TranscriptSegment],
    min_dur: float = 15.0,
    max_dur: float = 90.0
) -> List[ClipCandidate]:
    """
    Fallback clip finder when LLM is unavailable or yields no output.
    Uses sentence boundary analysis and heuristic scoring.
    """
    if not segments:
        return []

    candidates: List[ClipCandidate] = []
    HOOK_KEYWORDS = ["secret", "mistake", "truth", "never", "why", "how", "best", "worst", "imagine", "stop"]

    for i in range(len(segments)):
        for j in range(i, len(segments)):
            dur = segments[j].end - segments[i].start
            if dur > max_dur:
                break
            if dur < min_dur:
                continue

            text = " ".join(s.text for s in segments[i:j+1])
            first_sentence = segments[i].text.lower()
            
            # Heuristic score calculation
            score = 65
            has_hook = any(kw in first_sentence for kw in HOOK_KEYWORDS)
            if has_hook:
                score += 15
            if "?" in first_sentence:
                score += 10
            if "!" in text:
                score += 5
            score = min(98, score)

            # Heuristic breakdown metrics (0-10)
            hook_s = round(min(10.0, max(2.0, 6.0 + (2.5 if has_hook else 0.0) + (1.5 if "?" in first_sentence else 0.0))), 1)
            eng_s = round(min(10.0, max(2.0, 6.0 + (1.5 if "!" in text else 0.0) + (1.0 if has_hook else 0.0))), 1)
            info_s = round(min(10.0, max(2.0, 7.0 + (1.0 if any(c.isdigit() for c in text) else 0.0))), 1)
            emo_s = round(min(10.0, max(2.0, 5.5 + (2.0 if any(w in text.lower() for w in ["love", "hate", "crazy", "insane", "amazing"]) else 0.0))), 1)
            clar_s = round(min(10.0, max(2.0, 8.5 - (1.5 if dur > 60 else 0.0))), 1)
            stand_s = round(min(10.0, max(2.0, 8.5 - (2.5 if first_sentence.startswith(("and", "but", "so", "because")) else 0.0))), 1)
            
            breakdown = {
                "hook": hook_s,
                "engagement": eng_s,
                "info_value": info_s,
                "emotion": emo_s,
                "clarity": clar_s,
                "standalone": stand_s
            }

            candidates.append(ClipCandidate(
                title=f"Clip {len(candidates)+1}: {segments[i].text[:45]}...",
                start_time=seconds_to_timestamp(segments[i].start),
                end_time=seconds_to_timestamp(segments[j].end),
                start_sec=segments[i].start,
                end_sec=segments[j].end,
                duration=round(dur, 1),
                score=score,
                topic="Highlighted Moment",
                hook=segments[i].text,
                reason="Heuristically selected segment with clear sentence boundaries and active delivery.",
                context_required=False,
                transcript_snippet=text,
                breakdown=breakdown
            ))


    # Sort candidates by score descending and return top non-overlapping clips
    candidates.sort(key=lambda c: c.score, reverse=True)
    
    # Simple overlap filter
    filtered: List[ClipCandidate] = []
    for c in candidates:
        overlap = False
        for f in filtered:
            # Check overlap percentage
            inter = max(0.0, min(c.end_sec, f.end_sec) - max(c.start_sec, f.start_sec))
            if inter > 0.3 * min(c.duration, f.duration):
                overlap = True
                break
        if not overlap:
            filtered.append(c)
            if len(filtered) >= 6:
                break

    return filtered

def attach_snippets_to_clips(clips: List[ClipCandidate], segments: List[TranscriptSegment]):
    """Attach matching transcript segment text to candidates."""
    for clip in clips:
        matching_texts = [
            s.text for s in segments
            if (s.start >= clip.start_sec - 1.0) and (s.end <= clip.end_sec + 1.0)
        ]
        if matching_texts:
            clip.transcript_snippet = " ".join(matching_texts)
        elif not clip.transcript_snippet:
            clip.transcript_snippet = "Transcript segment not available."

def analyze_transcript_for_clips(
    segments: List[TranscriptSegment],
    timestamped_transcript_str: str,
    model_name: str = None
) -> List[ClipCandidate]:
    """
    Main clip analysis workflow:
    1. Try local Ollama analysis using LangChain Tool Calling.
    2. Fallback to heuristics if Ollama is offline or returns invalid JSON.
    """
    is_running, _, _ = check_ollama_status()
    
    if is_running:
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.prompts import ChatPromptTemplate
            from app.config import OLLAMA_BASE_URL

            # Setup LangChain ChatOllama with Tool Calling
            llm = ChatOllama(
                model=model_name or "llama3.2",
                base_url=OLLAMA_BASE_URL,
                temperature=0.2,
            )
            structured_llm = llm.with_structured_output(ClipAnalysisResponse)

            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("user", "Analyze the following timestamped transcript and extract the top 3-8 best YouTube Shorts / TikTok clip recommendations.\n\nTIMESTAMPED TRANSCRIPT:\n{transcript}\n\nRespond strictly with valid JSON conforming to the schema.")
            ])

            chain = prompt | structured_llm
            response = chain.invoke({"transcript": timestamped_transcript_str})

            if response and response.clips:
                clips = response.clips
                # Post-process timestamps
                for c in clips:
                    c.start_sec = timestamp_to_seconds(c.start_time)
                    c.end_sec = timestamp_to_seconds(c.end_time)
                    c.duration = round(max(0.0, c.end_sec - c.start_sec), 1)

                # Deduplicate and sort
                clips.sort(key=lambda c: c.score, reverse=True)
                attach_snippets_to_clips(clips, segments)
                return clips
        except Exception as e:
            print(f"[Warning] Ollama LangChain clip analysis failed: {e}. Falling back to heuristics.")

    # Fallback if Ollama unavailable or parsing returned 0 clips
    clips = fallback_heuristic_analyzer(segments)
    attach_snippets_to_clips(clips, segments)
    return clips
