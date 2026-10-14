# Feedback: Nebius Token Factory — from building "Feature Scientist"

*Draft for the Devpost "feedback on Token Factory / tools" writeup. Review before submitting.*

**Project:** Feature Scientist (Coding & Agentic Engineering track, Nebius x NVIDIA Global AI Hackathon).
Uses Nemotron as the hypothesis-generation backend inside an autonomous
feature-discovery loop: Nemotron proposes features, a critic/audit gate
judges them, and 3-seed ablations decide. Model:
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` via the Token Factory OpenAI-compatible API.

## What worked

- Key provisioning was fast and the key validated on first use. Vault-style
  storage from our side worked fine.
- `GET /v1/models` authenticated cleanly — that endpoint was reliable enough
  to become our credential-vs-network diagnostic.
- When calls went through, quality was real: our first Nemotron-backed run
  produced 2 hypotheses that the agent's critic rejected on legitimate
  statistical grounds (not on formatting) — exactly the failure mode we
  wanted to see. A later 3-round run produced 2 more live proposals that
  survived codegen and only died at our own 3-seed honesty gate. That's the
  Nemotron-in-the-loop footage for our demo video.

## What didn't

The single biggest friction was a full day of 403s on `POST /v1/chat/completions`
while `GET /v1/models` worked fine with the same credential. Debugging
progression, all from our sandboxed runtime:

1. First read: maybe the credential is wrong — but the models endpoint
   authenticated, so no.
2. Keyless POST returned `403 {"error":"policy_denied","policy":
   "hitl_domain_allow_nebius_com_8d025889"}` — pointing at our runtime's
   egress policy, not the key.
3. Later that evening, identical calls succeeded for reasons we could never
   establish. Flaky + opaque is worse than consistently blocked: we couldn't
   tell whether to keep debugging, route around, or wait.

We eventually fell back to a deterministic scripted backend for the science
runs and kept Nemotron for proposal generation only — a hybrid ("Nemotron
suggests, deterministic code implements") that worked well and is arguably
the better architecture anyway.

## Suggestions for the Token Factory team

1. **Document the sandbox story.** Agentic coding tools increasingly run in
   sandboxed runtimes with egress policies. A page on "calling Token Factory
   from sandboxed/CI environments" — which endpoints are POSTs, what error
   signatures indicate egress blocking vs auth failure — would have saved us
   most of a day.
2. **Make `policy_denied` actionable.** A policy name like
   `hitl_domain_allow_nebius_com_8d025889` with no remediation pointer is a
   dead end. Even a docs link in the error payload would help.
3. **Keep the error contract symmetric.** An auth-OK models endpoint next to
   a blocked chat-completions endpoint sent us chasing the credential first.
   If the two endpoints sit behind different policies, say so in the docs.
4. **Flakiness transparency.** Calls succeeding in the evening after failing
   all afternoon, with no status change we could see, is hard to build on.
   A lightweight status/incident note for the chat endpoint would be enough.

Net: the API itself was good once reachable; the pain was entirely in
diagnosing *why it wasn't reachable*. For an agentic-engineering hackathon,
where the build environment is itself part of the story, that diagnostic
gap is the thing to close.
