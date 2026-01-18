SAMPLE_BATCH = [
    {
        "category": "rag_overload",
        "query": "What are the key risks and required controls for the payment rollout?",
        "text": """
Background: Payment API launch for EU and US merchants next quarter. Compliance and reliability are the top constraints.

Risk: duplicate charge bugs seen in prior beta. Impact: double billing, refunds, support overload. Control: require idempotency keys, run replay tests on critical flows, and block deploys if error rate >0.5%.

Risk: PCI scope creep. Some engineers are logging full PAN data in debug traces. Control: scrub logs, ship automated detectors, rotate secrets weekly, and gate any PCI adjacent change behind security review.

Risk: regional downtime. EU region had 14 minutes of downtime last month. Control: add health probes per region, include failover runbooks, and simulate region failover every sprint.

Risk: slow dispute handling. Support team needs a checklist for high-risk transactions. Control: route transactions over $5,000 to manual review and require supervisor sign-off.

Note: Marketing wants to highlight speed. Constraint: never sacrifice auditability. Keep evidence for every decision in the risk register.
        """,
    },
    {
        "category": "meeting_transcript",
        "query": "Summarize the decisions and owners from the meeting",
        "text": """
Moderator: Today's goals are to decide MVP scope and lock owners. We keep mentioning the same blockers so let's capture them cleanly.

Alice: The data connector repeats the same schema mapping note three times in docs; we only need the concise version. She will trim the copy and publish a sandbox endpoint by Friday.

Ben: Latency is the user-visible issue. He proposes caching embeddings for repeated spans and wants a quick synthetic benchmark to prove savings.

Chandra: Concerned about deleting rare facts. Suggests boosting spans with dates and counts so we do not lose SLAs. She will own an ablation chart with quality vs compression ratio.

Decision: MVP includes greedy facility location, novelty boost, entity protection, baseline comparisons, and one click demo page. Owners: Alice (docs), Ben (benchmark), Chandra (analysis), Dana (frontend polish).

Action: share demo recording and deploy to staging with a feature flag for compression.
        """,
    },
    {
        "category": "policy_doc",
        "query": "What onboarding requirements must a contractor follow?",
        "text": """
Section: Identity and access
Contractors must use single sign-on with hardware keys. Shared accounts are prohibited. Access requests require manager approval and expire after 90 days.

Section: Devices
Only managed laptops may access production data. Machines need full disk encryption and monthly patching. USB storage is blocked by default.

Section: Data handling
Sensitive data must stay within approved storage buckets. Never email secrets. Audit logs must be retained for one year. Incident reports are due within 24 hours.

Section: Offboarding
All accounts are revoked on end date. Hardware must be returned within seven days. Badge access is disabled immediately. Managers confirm data handoff.
        """,
    },
]
