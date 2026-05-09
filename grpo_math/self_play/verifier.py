from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


class VerifierVerdict(str, Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    INVALID = "INVALID"
    UNCLEAR = "UNCLEAR"


_VERDICT_RE = re.compile(
    r"^\s*VERDICT\s*:\s*(CORRECT|INCORRECT|INVALID|UNCLEAR)\b",
    flags=re.IGNORECASE | re.MULTILINE,
)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class VerificationRequest:
    question: str
    completion: str


@dataclass(frozen=True)
class VerificationResult:
    verdict: VerifierVerdict
    raw_text: str
    confidence: float | None = None
    cached: bool = False
    model: str = ""
    prompt_version: str = ""
    finish_reason: str = ""
    usage: Mapping[str, Any] | None = None


class Verifier(Protocol):
    model: str
    prompt_version: str

    def verify(self, question: str, completion: str) -> VerificationResult:
        ...

    def verify_batch(self, requests: Sequence[VerificationRequest]) -> list[VerificationResult]:
        ...


def parse_verifier_response(text: str) -> tuple[VerifierVerdict, float | None]:
    matches = list(_VERDICT_RE.finditer(text))
    if not matches:
        return VerifierVerdict.UNCLEAR, None
    match = matches[-1]
    verdict = VerifierVerdict(match.group(1).upper())
    confidence_match = _CONFIDENCE_RE.search(text)
    confidence = float(confidence_match.group(1)) if confidence_match else None
    return verdict, confidence


def render_verifier_prompt(template: str, question: str, completion: str) -> str:
    candidate_answer = extract_candidate_final_answer(completion) or "NONE"
    return template.format(
        question=question,
        solution=completion,
        completion=completion,
        candidate_answer=candidate_answer,
    )


def extract_candidate_final_answer(completion: str) -> str | None:
    search_text = completion
    if "[Final]" in completion:
        search_text = completion.rsplit("[Final]", 1)[1]
    final_matches = re.findall(r"FINAL_ANSWER\s*:\s*(.+?)(?:\n|$)", search_text, flags=re.IGNORECASE)
    if not final_matches:
        return None
    answer = final_matches[-1].strip()
    if answer.lower().strip(" .") in {"<final answer>", "final answer"}:
        return None
    return answer or None


def cache_key(
    *,
    question: str,
    completion: str,
    prompt_version: str,
    model: str,
) -> str:
    payload = {
        "question": question,
        "completion": completion,
        "prompt_version": prompt_version,
        "model": model,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class VerifierCache:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._entries: dict[str, VerificationResult] = {}
        if self.path is not None and self.path.exists():
            self._load()

    def get(self, key: str) -> VerificationResult | None:
        return self._entries.get(key)

    def set(self, key: str, result: VerificationResult) -> None:
        self._entries[key] = result
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "key": key,
                "verdict": result.verdict.value,
                "raw_text": result.raw_text,
                "confidence": result.confidence,
                "model": result.model,
                "prompt_version": result.prompt_version,
            }
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            self._entries[str(row["key"])] = VerificationResult(
                verdict=VerifierVerdict(str(row["verdict"]).upper()),
                raw_text=str(row.get("raw_text", "")),
                confidence=row.get("confidence"),
                model=str(row.get("model", "")),
                prompt_version=str(row.get("prompt_version", "")),
            )


class BaseVerifier:
    model: str
    prompt_version: str

    def verify_batch(self, requests: Sequence[VerificationRequest]) -> list[VerificationResult]:
        return [self.verify(req.question, req.completion) for req in requests]


class FixtureVerifier(BaseVerifier):
    def __init__(
        self,
        fixtures: Mapping[tuple[str, str], str | VerifierVerdict | VerificationResult],
        *,
        model: str = "fixture-verifier",
        prompt_version: str = "fixture-v1",
    ) -> None:
        self.fixtures = dict(fixtures)
        self.model = model
        self.prompt_version = prompt_version

    def verify(self, question: str, completion: str) -> VerificationResult:
        try:
            fixture = self.fixtures[(question, completion)]
        except KeyError as exc:
            raise KeyError("Missing fixture verifier result for question/completion pair.") from exc
        if isinstance(fixture, VerificationResult):
            return fixture
        if isinstance(fixture, VerifierVerdict):
            verdict, confidence, raw_text = fixture, None, f"VERDICT: {fixture.value}"
        else:
            verdict, confidence = parse_verifier_response(fixture)
            raw_text = fixture
        return VerificationResult(
            verdict=verdict,
            raw_text=raw_text,
            confidence=confidence,
            model=self.model,
            prompt_version=self.prompt_version,
        )


class CachedVerifier(BaseVerifier):
    def __init__(self, verifier: Verifier, cache: VerifierCache) -> None:
        self.verifier = verifier
        self.cache = cache
        self.model = verifier.model
        self.prompt_version = verifier.prompt_version

    def verify(self, question: str, completion: str) -> VerificationResult:
        key = cache_key(
            question=question,
            completion=completion,
            prompt_version=self.prompt_version,
            model=self.model,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return VerificationResult(
                verdict=cached.verdict,
                raw_text=cached.raw_text,
                confidence=cached.confidence,
                cached=True,
                model=cached.model,
                prompt_version=cached.prompt_version,
            )
        result = self.verifier.verify(question, completion)
        self.cache.set(key, result)
        return result


class PythonAssistedVerifier(BaseVerifier):
    """Attach a deterministic local Python check for verifiers that can use tool output."""

    def __init__(self, verifier: Verifier, *, timeout_s: float = 5.0) -> None:
        self.verifier = verifier
        self.timeout_s = timeout_s
        self.model = verifier.model
        self.prompt_version = verifier.prompt_version

    def verify(self, question: str, completion: str) -> VerificationResult:
        tool_result = run_python_verification_probe(question, completion, timeout_s=self.timeout_s)
        augmented_question = f"{question}\n\nPython tool result:\n{tool_result}"
        return self.verifier.verify(augmented_question, completion)

    def verify_batch(self, requests: Sequence[VerificationRequest]) -> list[VerificationResult]:
        return [self.verify(req.question, req.completion) for req in requests]


def run_python_verification_probe(question: str, completion: str, *, timeout_s: float = 5.0) -> str:
    final_answer = extract_candidate_final_answer(completion)
    script = (
        "from fractions import Fraction\n"
        "import json, math, re\n"
        f"question = {question!r}\n"
        f"completion = {completion!r}\n"
        f"candidate_final_answer = {final_answer!r}\n"
        "print('candidate_final_answer =', candidate_final_answer)\n"
        "ints = [int(x) for x in re.findall(r'-?\\d+', question)]\n"
        "print('integers_in_question =', json.dumps(ints))\n"
        "if candidate_final_answer is not None:\n"
        "    s = str(candidate_final_answer).strip()\n"
        "    try:\n"
        "        print('candidate_as_int =', int(s))\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        safe = s.replace('^', '**')\n"
        "        if re.fullmatch(r'[0-9\\s\\*\\+\\-\\/\\(\\)\\.]+', safe):\n"
        "            value = eval(safe, {'__builtins__': {}}, {})\n"
        "            if isinstance(value, float) and value.is_integer():\n"
        "                value = int(value)\n"
        "            if isinstance(value, int):\n"
        "                print('candidate_as_expression_int =', value)\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        print('candidate_as_fraction =', str(Fraction(s)))\n"
        "    except Exception:\n"
        "        pass\n"
        "\n"
        "def emit_exact(label, value):\n"
        "    print(label, '=', value)\n"
        "    if candidate_final_answer is not None:\n"
        "        try:\n"
        "            print('candidate_matches_exact =', str(Fraction(str(candidate_final_answer).strip()) == Fraction(value)))\n"
        "        except Exception:\n"
        "            print('candidate_matches_exact =', str(str(candidate_final_answer).strip() == str(value)))\n"
        "\n"
        "q_lower = question.lower()\n"
        "if 'x^2 + y^2' in question and 'gcd' in q_lower and 'f(100)' in question:\n"
        "    count = 0\n"
        "    for x in range(1, 101):\n"
        "        for y in range(1, 101):\n"
        "            if x*x + y*y == 100*100 and math.gcd(x, y) == 1:\n"
        "                count += 1\n"
        "    emit_exact('exact_expected_answer', count)\n"
        "if 'ends in the digits' in q_lower and 'n^3' in question:\n"
        "    nums = [int(x) for x in re.findall(r'\\d+', question)]\n"
        "    if len(nums) >= 2:\n"
        "        limit, suffix = max(nums), nums[-1]\n"
        "        mod = 10 ** len(str(suffix))\n"
        "        count = sum(1 for n in range(1, limit + 1) if pow(n, 3, mod) == suffix)\n"
        "        emit_exact('exact_expected_answer', count)\n"
        "if 'sum of' in q_lower and 'digits' in q_lower and 'divisible by' in q_lower:\n"
        "    nums = [int(x) for x in re.findall(r'\\d+', question)]\n"
        "    if nums:\n"
        "        limit = None\n"
        "        if 'less than' in q_lower and len(nums) >= 1:\n"
        "            limit = nums[0] - 1\n"
        "        elif 'at most' in q_lower and len(nums) >= 1:\n"
        "            limit = nums[0]\n"
        "        modulus = nums[-1]\n"
        "        divisor = nums[-2] if len(nums) >= 2 else modulus\n"
        "        if limit is not None and limit > 0 and divisor != 0 and modulus != 0:\n"
        "            vals = [n for n in range(1, limit + 1) if n % divisor == 0 and sum(int(ch) for ch in str(abs(n))) % modulus == 0]\n"
        "            print('python_enumerated_values =', json.dumps(vals[:100]))\n"
        "            print('python_enumerated_count =', len(vals))\n"
        "            if any(word in q_lower for word in ['how many', 'number of elements', 'count']):\n"
        "                emit_exact('exact_expected_answer', len(vals))\n"
        "            elif 'sum of all' in q_lower or 'find the sum' in q_lower:\n"
        "                emit_exact('exact_expected_answer', sum(vals))\n"
        "\n"
        "def divisor_count(x):\n"
        "    if x <= 0:\n"
        "        return 0\n"
        "    total = 1\n"
        "    d = 2\n"
        "    while d * d <= x:\n"
        "        exp = 0\n"
        "        while x % d == 0:\n"
        "            x //= d\n"
        "            exp += 1\n"
        "        if exp:\n"
        "            total *= exp + 1\n"
        "        d += 1 if d == 2 else 2\n"
        "    if x > 1:\n"
        "        total *= 2\n"
        "    return total\n"
        "\n"
        "def parse_candidate_int(s):\n"
        "    if s is None:\n"
        "        return None\n"
        "    s = str(s).strip().replace('^', '**')\n"
        "    try:\n"
        "        if re.fullmatch(r'[0-9\\s\\*\\+\\-\\/\\(\\)\\.]+', s):\n"
        "            value = eval(s, {'__builtins__': {}}, {})\n"
        "            if isinstance(value, float) and value.is_integer():\n"
        "                value = int(value)\n"
        "            if isinstance(value, int):\n"
        "                return value\n"
        "    except Exception:\n"
        "        return None\n"
        "    return None\n"
        "\n"
        "if ('number of divisors' in q_lower or 'd(n' in q_lower) and 'n^2' in question and '12' in question:\n"
        "    cand_n = parse_candidate_int(candidate_final_answer)\n"
        "    if cand_n is not None:\n"
        "        dn = divisor_count(cand_n)\n"
        "        dn2 = divisor_count(cand_n * cand_n)\n"
        "        print('candidate_n =', cand_n)\n"
        "        print('candidate_divisor_count_n =', dn)\n"
        "        print('candidate_divisor_count_n_squared =', dn2)\n"
        "        print('candidate_satisfies_divisor_ratio =', str(dn2 == 12 * dn))\n"
        "    found = None\n"
        "    for n in range(1, 200000):\n"
        "        if divisor_count(n * n) == 12 * divisor_count(n):\n"
        "            found = n\n"
        "            break\n"
        "    if found is not None:\n"
        "        emit_exact('exact_expected_answer', found)\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return "Python probe timed out."
    output = completed.stdout.strip()
    if completed.stderr.strip():
        output = f"{output}\nstderr: {completed.stderr.strip()}".strip()
    return output or "Python probe produced no output."


class OpenAICompatibleVerifier(BaseVerifier):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        prompt_template: str,
        prompt_version: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 32,
        max_tokens_param: str = "max_completion_tokens",
        extra_body: Mapping[str, Any] | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 6,
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        request_fn: Callable[[str, Mapping[str, Any], Mapping[str, str], float], Mapping[str, Any]]
        | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.prompt_template = prompt_template
        self.prompt_version = prompt_version
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.max_tokens_param = max_tokens_param
        self.extra_body = dict(extra_body or {})
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.initial_backoff_s = initial_backoff_s
        self.max_backoff_s = max_backoff_s
        self.sleep_fn = sleep_fn
        self.request_fn = request_fn or _post_json

    def verify(self, question: str, completion: str) -> VerificationResult:
        prompt = render_verifier_prompt(self.prompt_template, question, completion)
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        payload[self.max_tokens_param] = self.max_completion_tokens
        payload.update(self.extra_body)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "self-play-verifier/1.0",
        }
        body = self._request_with_retries(payload, headers)
        raw_text = extract_chat_content(body)
        verdict, confidence = parse_verifier_response(raw_text)
        return VerificationResult(
            verdict=verdict,
            raw_text=raw_text,
            confidence=confidence,
            model=self.model,
            prompt_version=self.prompt_version,
            finish_reason=extract_finish_reason(body),
            usage=extract_usage(body),
        )

    def _request_with_retries(
        self, payload: Mapping[str, Any], headers: Mapping[str, str]
    ) -> Mapping[str, Any]:
        url = self.base_url.rstrip("/") + "/chat/completions"
        attempt = 0
        backoff_s = max(0.0, self.initial_backoff_s)
        while True:
            attempt += 1
            try:
                return self.request_fn(url, payload, headers, self.timeout_s)
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt > self.max_retries:
                    raise
            except Exception:
                if attempt > self.max_retries:
                    raise
            self.sleep_fn(backoff_s)
            backoff_s = min(backoff_s * 2.0, self.max_backoff_s)


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_s: float,
) -> Mapping[str, Any]:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp is not None else ""
        if body:
            raise urllib.error.HTTPError(
                exc.url,
                exc.code,
                f"{exc.reason}: {body}",
                exc.headers,
                None,
            ) from exc
        raise


def extract_chat_content(body: Mapping[str, Any]) -> str:
    try:
        message = body["choices"][0]["message"]  # type: ignore[index]
    except Exception as exc:
        raise KeyError(f"Missing choices/message in verifier response: {exc}") from exc
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        if parts:
            return "\n".join(parts).strip()
    # Do not treat provider-specific hidden reasoning as the model answer.
    # Thinking models should be configured to return final content explicitly.
    text = message.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    raise KeyError("Could not parse verifier response text.")


def extract_finish_reason(body: Mapping[str, Any]) -> str:
    try:
        reason = body["choices"][0].get("finish_reason")  # type: ignore[index, union-attr]
    except Exception:
        return ""
    return str(reason) if reason is not None else ""


def extract_usage(body: Mapping[str, Any]) -> Mapping[str, Any] | None:
    usage = body.get("usage")
    return usage if isinstance(usage, Mapping) else None

