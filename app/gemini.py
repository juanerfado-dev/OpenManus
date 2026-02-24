import os
import re
from typing import List, Dict, Any

GEMINI_ERROR_INVALID_FORMAT = "Invalid Gemini response format"
GEMINI_ERROR_NETWORK = "Gemini network error"
GEMINI_ERROR_API = "Gemini API error"

class AsyncGeminiClient:
    def __init__(self, *args, **kwargs):
        # Inicialização real deve ser feita conforme o restante do projeto
        pass

    async def _create(self, *args, stream: bool = True, **kwargs):
        """
        Executa chamada Gemini e normaliza resposta.

        Retorno:
            - Sempre retorna string normalizada (nunca None)
            - Em caso de erro de parsing/estrutura: GEMINI_ERROR_INVALID_FORMAT
            - Em caso de erro de rede: GEMINI_ERROR_NETWORK (permite fallback automático)
            - Em caso de erro HTTP/API: GEMINI_ERROR_API (permite fallback opcional)
            - Resposta válida: texto normalizado

        Pontos de fallback:
            - Network error: fallback automático
            - API error: fallback opcional
            - Format error: não fazer fallback automático
        """
        try:
            # Exemplo mínimo de chamada (ajuste conforme integração real):
            # Montagem do prompt e payload
            messages: List[Dict[str, Any]] = kwargs.get("messages", []) or []
            prompt = ""
            if messages:
                prompt = " ".join([m.get("content", "") for m in messages])
            # Simulação de chamada HTTP (substitua pelo real)
            if not prompt:
                raise ValueError("Prompt vazio")
            # Suponha resposta válida
            return prompt.strip()
        except (TimeoutError, OSError):
            return GEMINI_ERROR_NETWORK
        except ValueError:
            return GEMINI_ERROR_INVALID_FORMAT
        except Exception:
            return GEMINI_ERROR_API
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        # simple in-memory cache of last prompt -> response
        self._last_prompt: Optional[str] = None
        self._last_response: Optional[str] = None
        # persistent cache (file or redis)
        self._persistent_cache = PersistentCache()

    async def close(self):
        await self._client.aclose()

    def _count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text))

    def _count_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        # rough estimation compatible with project's TokenCounter
        total = 2
        for m in messages:
            total += 4
            total += self._count_text_tokens(m.get("role", ""))
            content = m.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        total += self._count_text_tokens(item)
                    elif isinstance(item, dict) and "text" in item:
                        total += self._count_text_tokens(item["text"])
            elif isinstance(content, str):
                total += self._count_text_tokens(content)
        return total

    def _to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                text = " ".join([i if isinstance(i, str) else i.get("text", "") for i in content])
            else:
                text = content or ""
            parts.append(f"[{role}] {text}")
        return "\n".join(parts)

    def _summarize_texts(self, texts: List[str], max_lines: int = 5) -> str:
        # Simple extractive summarization: take first sentence from each text until limit
        summary_lines = []
        for t in texts:
            t = (t or "").strip()
            try:
                if isinstance(text, dict):
                    if "parts" in text and isinstance(text["parts"], list):
                        text = " ".join(part.get("text", "") for part in text.get("parts", []))
                    elif "text" in text:
                        text = text["text"]
                    elif "content" in text:
                        text = text["content"]
                    else:
                        import json
                        text = json.dumps(text)
                elif isinstance(text, list):
                    text = " ".join([str(t.get("text", t)) if isinstance(t, dict) else str(t) for t in text])
                elif not isinstance(text, str):
                    text = str(text)
                clean_text = re.sub(r"\s+", " ", (text or "")).strip()
            except Exception:
                logger.debug("Invalid Gemini response format (post-parsing)")
                return GEMINI_ERROR_INVALID_FORMAT
            return clean_text

        # Ensure messages is a list of dicts
        if not isinstance(messages, list):
            messages = [messages]

        # Keep system messages always
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        # Default keep last N messages (3 interactions = 6 messages)
        KEEP_LAST = 6
        # If messages length exceeds KEEP_LAST, trim older
        if len(other_msgs) > KEEP_LAST:
            other_msgs = other_msgs[-KEEP_LAST:]

        # Ensure system prompts are not long: replace long system prompts with a fixed short system prompt
        FIXED_SYSTEM_PROMPT = "Você é um assistente conciso, objetivo e informativo."  # <200 chars
        new_system_msgs = []
        for sm in system_msgs:
            content = sm.get("content", "")
            if len((content or "")) > 200:
                new_system_msgs = [{"role": "system", "content": FIXED_SYSTEM_PROMPT}]
                break
            else:
                new_system_msgs.append(sm)

        # Use new_system_msgs for final assembly
        assembled = new_system_msgs + other_msgs

        # Build prompt and collapse whitespace
        prompt = self._to_prompt(assembled)
        prompt = re.sub(r"\s+", " ", prompt).strip()

        # SAFE limits
        SAFE_CHAR_LIMIT = 4000
        SUMMARY_CHAR_LIMIT = 500

        summarized_flag = False

        # If prompt too large, summarize older messages but keep last user message intact
        if len(prompt) > SAFE_CHAR_LIMIT:
            summarized_flag = True
            # find last user message in other_msgs (after trimming to KEEP_LAST)
            last_user_idx = None
            for i in range(len(other_msgs) - 1, -1, -1):
                if other_msgs[i].get("role") == "user":
                    last_user_idx = i
                    break

            if last_user_idx is None:
                # no user message, fallback to keeping last message
                last_user_idx = len(other_msgs) - 1

            to_summarize = other_msgs[:last_user_idx]
            keep_after = other_msgs[last_user_idx:]

            # Collect texts to summarize
            texts = []
            for m in to_summarize:
                content = m.get("content", "")
                if isinstance(content, list):
                    texts.append(" ".join([i if isinstance(i, str) else i.get("text", "") for i in content]))
                else:
                    texts.append(content or "")

            # Create a compact summary up to SUMMARY_CHAR_LIMIT
            summary = self._summarize_texts(texts, max_lines=10)
            if len(summary) > SUMMARY_CHAR_LIMIT:
                summary = summary[:SUMMARY_CHAR_LIMIT]

            summary_msg = {"role": "system", "content": f"Resumo do histórico: {summary}"}
            final_messages = new_system_msgs + [summary_msg] + keep_after
            prompt = self._to_prompt(final_messages)
            prompt = re.sub(r"\s+", " ", prompt).strip()
        else:
            final_messages = assembled

        # Call Gemini REST API (non-streaming) - Google AI Studio v1beta generateContent
        url = f"{self.base_url}/v1beta/models/{model}:generateContent"
        # Build payload per required format
        # Minimal payload accepted by generateContent; avoid sending unknown top-level fields
        body = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }

        # API key must be set in environment only
        if not self.api_key:
            logger.error("GEMINI_API_KEY not set in environment")
            raise RuntimeError("GEMINI_API_KEY not set")

        url = f"{url}?key={self.api_key}"
        try:
            logger.info(f"Gemini request chars={len(prompt)} model={model}")
        except Exception:
            pass

        # Cache: if identical prompt to last, return cached response without calling API
        if hasattr(self, "_last_prompt") and self._last_prompt == prompt and self._last_response is not None:
            logger.info("Using cached Gemini response")
            clean_text = self._last_response
            prompt_tokens = self._count_messages_tokens(final_messages)
            completion_tokens = self._count_text_tokens(clean_text)
            return _NonStreamResponse(clean_text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

        # Use a simple call; for streaming we emulate by splitting the returned text

        # --- Gemini provider block ---
        text = ""
        try:
            try:
                r = await self._client.post(url, json=body)
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.NetworkError, httpx.TransportError, httpx.TimeoutException, OSError) as net_exc:
                # Erro de rede/timeout/transporte
                logger.error(f"Gemini network error: {net_exc}")
                # Aqui pode-se acionar fallback automático
                raise RuntimeError("Gemini network error")
            except Exception as net_exc:
                # Outros erros de rede
                logger.error(f"Gemini network error: {net_exc}")
                # Aqui pode-se acionar fallback automático
                raise RuntimeError("Gemini network error")

            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as he:
                logger.error(f"Gemini API error: {he}")
                # Aqui pode-se acionar fallback opcional
                raise RuntimeError("Gemini API error")

            try:
                data = r.json()
            except Exception as parse_exc:
                logger.error(f"Gemini response parse error: {parse_exc}")
                raise RuntimeError("Invalid Gemini response format")

            # attempt to extract text from known fields
            if isinstance(data, dict):
                # Try multiple possible response shapes
                # 1) candidates -> output
                if data.get("candidates"):
                    cand = data["candidates"][0]
                    text = cand.get("output") or cand.get("content") or str(cand)
                # 2) outputs -> content -> parts -> text (newer generateContent shape)
                elif data.get("outputs"):
                    try:
                        out0 = data["outputs"][0]
                        cont = out0.get("content") or out0.get("contents")
                        if isinstance(cont, list) and cont:
                            first = cont[0]
                            parts = first.get("parts") or first.get("text")
                            if parts and isinstance(parts, list):
                                texts = []
                                for p in parts:
                                    if isinstance(p, dict):
                                        texts.append(p.get("text") or p.get("content") or "")
                                    elif isinstance(p, str):
                                        texts.append(p)
                                text = " ".join([t for t in texts if t])
                            elif isinstance(first, dict) and first.get("text"):
                                text = first.get("text")
                            else:
                                text = str(first)
                        else:
                            text = str(out0)
                    except Exception:
                        text = str(data)
                elif data.get("output"):
                    out = data.get("output")
                    if isinstance(out, dict):
                        text = out.get("text") or out.get("content") or str(out)
                    else:
                        text = out
                else:
                    # fallback to stringifying
                    text = str(data)
            else:
                text = str(data)
        except RuntimeError as e:
            # Erro já normalizado acima
            raise
        except Exception as parse_exc:
            logger.error(f"Gemini response format error: {parse_exc}")
            # Aqui NÃO deve acionar fallback automático
            raise RuntimeError("Invalid Gemini response format")


        # Extração robusta e normalização final
        try:
            if isinstance(text, dict):
                if "parts" in text and isinstance(text["parts"], list):
                    text = " ".join(part.get("text", "") for part in text.get("parts", []))
                elif "text" in text:
                    text = text["text"]
                elif "content" in text:
                    text = text["content"]
                else:
                    import json
                    text = json.dumps(text)
            elif isinstance(text, list):
                async def _create(self, *args, stream: bool = True, **kwargs):
                    """
                    Executa chamada Gemini e normaliza resposta.

                    Retorno:
                        - Sempre retorna string normalizada (nunca None)
                        - Em caso de erro de parsing/estrutura: GEMINI_ERROR_INVALID_FORMAT
                        - Em caso de erro de rede: GEMINI_ERROR_NETWORK (permite fallback automático)
                        - Em caso de erro HTTP/API: GEMINI_ERROR_API (permite fallback opcional)
                        - Resposta válida: texto normalizado

                    Pontos de fallback:
                        - Network error: fallback automático
                        - API error: fallback opcional
                        - Format error: não fazer fallback automático
                    """
                    messages: List[Dict[str, Any]] = kwargs.get("messages", []) or []
                    model = kwargs.get("model", self.DEFAULT_MODEL)

                    if "max_output_tokens" not in kwargs and "max_tokens" in kwargs:
                        kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
                    kwargs.setdefault("max_output_tokens", self.DEFAULT_MAX_OUTPUT_TOKENS)
                    kwargs.setdefault("temperature", self.DEFAULT_TEMPERATURE)
                    kwargs.setdefault("top_p", self.DEFAULT_TOP_P)

                    if not isinstance(messages, list):
                        messages = [messages]

                    system_msgs = [m for m in messages if m.get("role") == "system"]
                    other_msgs = [m for m in messages if m.get("role") != "system"]

                    KEEP_LAST = 6
                    if len(other_msgs) > KEEP_LAST:
                        other_msgs = other_msgs[-KEEP_LAST:]

                    FIXED_SYSTEM_PROMPT = "Você é um assistente conciso, objetivo e informativo."
                    new_system_msgs = []
                    for sm in system_msgs:
                        content = sm.get("content", "")
                        if len((content or "")) > 200:
                            new_system_msgs = [{"role": "system", "content": FIXED_SYSTEM_PROMPT}]
                            break
                        else:
                            new_system_msgs.append(sm)

                    assembled = new_system_msgs + other_msgs
                    prompt = self._to_prompt(assembled)
                    prompt = re.sub(r"\s+", " ", prompt).strip()

                    SAFE_CHAR_LIMIT = 4000
                    SUMMARY_CHAR_LIMIT = 500
                    summarized_flag = False
                    if len(prompt) > SAFE_CHAR_LIMIT:
                        summarized_flag = True
                        last_user_idx = None
                        for i in range(len(other_msgs) - 1, -1, -1):
                            if other_msgs[i].get("role") == "user":
                                last_user_idx = i
                                break
                        if last_user_idx is None:
                            last_user_idx = len(other_msgs) - 1
                        to_summarize = other_msgs[:last_user_idx]
                        keep_after = other_msgs[last_user_idx:]
                        texts = []
                        for m in to_summarize:
                            content = m.get("content", "")
                            if isinstance(content, list):
                                texts.append(" ".join([i if isinstance(i, str) else i.get("text", "") for i in content]))
                            else:
                                texts.append(content or "")
                        summary = self._summarize_texts(texts, max_lines=10)
                        if len(summary) > SUMMARY_CHAR_LIMIT:
                            summary = summary[:SUMMARY_CHAR_LIMIT]
                        summary_msg = {"role": "system", "content": f"Resumo do histórico: {summary}"}
                        final_messages = new_system_msgs + [summary_msg] + keep_after
                        prompt = self._to_prompt(final_messages)
                        prompt = re.sub(r"\s+", " ", prompt).strip()
                    else:
                        final_messages = assembled

                    url = f"{self.base_url}/v1beta/models/{model}:generateContent"
                    body = {
                        "contents": [
                            {"parts": [{"text": prompt}]}
                        ]
                    }

                    if not self.api_key:
                        logger.error("GEMINI_API_KEY not set in environment")
                        return GEMINI_ERROR_INVALID_FORMAT

                    url = f"{url}?key={self.api_key}"
                    try:
                        logger.info(f"Gemini request chars={len(prompt)} model={model}")
                    except Exception:
                        pass

                    if hasattr(self, "_last_prompt") and self._last_prompt == prompt and self._last_response is not None:
                        logger.info("Using cached Gemini response")
                        clean_text = self._last_response
                        prompt_tokens = self._count_messages_tokens(final_messages)
                        completion_tokens = self._count_text_tokens(clean_text)
                        return _NonStreamResponse(clean_text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

                    text = ""
                    try:
                        try:
                            r = await self._client.post(url, json=body)
                        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.NetworkError, httpx.TransportError, httpx.TimeoutException, OSError) as net_exc:
                            logger.warning(f"Gemini network error: {net_exc}")
                            # Fallback automático permitido aqui
                            return GEMINI_ERROR_NETWORK
                        except Exception as net_exc:
                            logger.warning(f"Gemini network error: {net_exc}")
                            # Fallback automático permitido aqui
                            return GEMINI_ERROR_NETWORK

                        try:
                            r.raise_for_status()
                        except httpx.HTTPStatusError as he:
                            logger.error(f"Gemini API error: {he}")
                            # Fallback opcional permitido aqui
                            return GEMINI_ERROR_API

                        try:
                            data = r.json()
                        except Exception as parse_exc:
                            logger.debug(f"Gemini response parse error: {parse_exc}")
                            return GEMINI_ERROR_INVALID_FORMAT

                        if isinstance(data, dict):
                            if data.get("candidates"):
                                cand = data["candidates"][0]
                                text = cand.get("output") or cand.get("content") or str(cand)
                            elif data.get("outputs"):
                                try:
                                    out0 = data["outputs"][0]
                                    cont = out0.get("content") or out0.get("contents")
                                    if isinstance(cont, list) and cont:
                                        first = cont[0]
                                        parts = first.get("parts") or first.get("text")
                                        if parts and isinstance(parts, list):
                                            texts = []
                                            for p in parts:
                                                if isinstance(p, dict):
                                                    texts.append(p.get("text") or p.get("content") or "")
                                                elif isinstance(p, str):
                                                    texts.append(p)
                                            text = " ".join([t for t in texts if t])
                                        elif isinstance(first, dict) and first.get("text"):
                                            text = first.get("text")
                                        else:
                                            text = str(first)
                                    else:
                                        text = str(out0)
                                except Exception:
                                    text = str(data)
                            elif data.get("output"):
                                out = data.get("output")
                                if isinstance(out, dict):
                                    text = out.get("text") or out.get("content") or str(out)
                                else:
                                    text = out
                            else:
                                text = str(data)
                        else:
                            text = str(data)
                    except Exception as parse_exc:
                        logger.debug(f"Gemini response format error: {parse_exc}")
                        # Não fazer fallback automático aqui
                        return GEMINI_ERROR_INVALID_FORMAT

                    try:
                        if isinstance(text, dict):
                            if "parts" in text and isinstance(text["parts"], list):
                                text = " ".join(part.get("text", "") for part in text.get("parts", []))
                            elif "text" in text:
                                text = text["text"]
                            elif "content" in text:
                                text = text["content"]
                            else:
                                import json
                                text = json.dumps(text)
                        elif isinstance(text, list):
                            text = " ".join([str(t.get("text", t)) if isinstance(t, dict) else str(t) for t in text])
                        elif not isinstance(text, str):
                            text = str(text)








