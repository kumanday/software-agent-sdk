## OpenHands + ChatGPT Codex Subscription: Integration Summary

We’ve been trying to integrate OpenHands with the ChatGPT Codex subscription backend, following the guidance to “reuse the Codex client.” Despite closely mirroring the OpenCode implementation, we’re consistently getting this error from the Codex Responses endpoint:

```json
{"detail": "Instructions are not valid"}
```

for `POST https://chatgpt.com/backend-api/codex/responses`.

### 1. OAuth / client identity

- We use the same values as OpenCode:
  - `CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"`
  - `ISSUER = "https://auth.openai.com"`
  - `CODEX_API_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"`
- Authorization code and refresh flows match:
  - `POST https://auth.openai.com/oauth/token` with
    - `grant_type=authorization_code` or `refresh_token`
    - `client_id=app_EMoamEEZ73f0CkXaXp7hrann`
    - same PKCE and redirect URI handling.

### 2. Headers and originator

- For the Codex `responses` call, we set:
  - `Authorization: Bearer <access_token>`
  - `originator: "opencode"` **or** `originator: "openhands"` (we have tried *both* values).
  - `User-Agent` is different in string value but similar in structure.

### 3. Model and endpoint

- We call the same backend as OpenCode:
  - `POST https://chatgpt.com/backend-api/codex/responses`
- With `model` set to, e.g., `"gpt-5.2-codex"`.

### 4. Instructions field

- From OpenCode, we captured the working payload from `~/.local/share/opencode/log/dev.log` and confirmed:
  - `instructions` is the full contents of `packages/opencode/src/session/prompt/codex_header.txt` (≈ 8.6k characters).
- In OpenHands we have tried **two** variants for `instructions` when using the subscription Codex transport:
  1. A short OpenHands-specific sentence derived from its default system prompt, e.g.
     - `"You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks."`
  2. The **exact** Codex header used by OpenCode (`codex_header.txt`, post-`trim()`), embedded as a long string constant.
- In both cases, we log the outgoing payload and see `instructions` set to the expected string.

### 5. Input payload shape

- From OpenCode’s logged payload, we saw `input` entries like:

  ```json
  { "role": "user", "content": [{ "type": "input_text", "text": "..." }] }
  ```

  (no `"type": "message"` wrapper).

- In OpenHands we changed the Responses formatter so that, for the subscription Codex transport:
  - `input` has the same structure:

    ```json
    [
      {
        "role": "user",
        "content": [
          { "type": "input_text", "text": "Say 'ok' and nothing else." }
        ]
      }
    ]
    ```

  - System messages are moved into the first user message as a prefixed `input_text` (similar in spirit to how OpenCode pushes system context into the prompt), and `store` is set to `false` for Codex.

### 6. Other options

- We disable `max_output_tokens` for Codex, matching OpenCode’s behavior.
- For the minimal repro we do **not** send tools or additional options:
  - Plain user prompt: `"Say 'ok' and nothing else."`
  - No tools, no extra response format.

### 7. Result

- OpenCode’s client succeeds against `https://chatgpt.com/backend-api/codex/responses` with this configuration.
- OpenHands, using combinations of:
  - the **same** `CLIENT_ID`,
  - `originator="opencode"` **and** `originator="openhands"` (both tested),
  - both the **OpenHands**-specific and the **OpenCode** Codex header as `instructions`,
  - the same `input` shape as OpenCode,

  continues to receive a 400 with:

  ```json
  {"detail": "Instructions are not valid"}
  ```

### 8. What we suspect and what we’re asking

Given that we’ve aligned all visible fields in the request (OAuth, headers, endpoint, model, `instructions` content, and `input` structure), but Codex still rejects our requests while accepting OpenCode’s, we suspect at least one of the following:

- There is an additional server-side allowlist or validation tied to more than just `CLIENT_ID + originator + instructions` (e.g., tied to a specific client identity or other internal metadata), or
- There is some subtle difference in how our request is interpreted by the Codex backend that is not obvious from the SDK layer.

We’d appreciate help either:

1. Confirming whether the Codex backend has additional checks that would treat OpenHands differently from the official Codex client, even when the wire payload is effectively the same, or
2. Providing the exact contract (including any hidden constraints) that requests must satisfy so we can make OpenHands fully compatible with the ChatGPT Codex subscription endpoint.
