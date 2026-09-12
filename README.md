# CLARITY-Unmasking-Political-Question-Evasions
An attempt at the SemEval 2026 Task


1. The organizers' scorer/task repo — clone first, non-negotiable. Your entire C4 contribution is a decision rule derived from the scorer's exact semantics. You currently have those semantics from TeleAI's Appendix A.1 — a competitor's prose paraphrase. If the real scorer differs in any detail (how macro-averaging interacts with the multi-reference TP/FN accounting, how classes with zero support are handled), your derivation is wrong and the whole contribution collapses. You need the actual source, and you need to reimplement it and unit-test your version against it.

2. TeleAI (ther7777/semeval-2026-task6-camsr-cot) — clone, high value. Their Appendix B.3 refined label definitions are already, in effect, a coverage-then-engagement decision tree ("Does the answer address the core target at all?" → "Does it provide ANY key information?" → "fully/partially"). That is independent, adversarial corroboration that the taxonomy has a latent coverage axis — which is exactly C2's claim, arrived at by a team that wasn't trying to prove it. Their mined confusion matrix tells you which boundary pairs actually matter, and their key-information override rule is a ready-made specification for your C5 span judgment.

3. ChulaNLP (moswisarut) — attempt it, don't block on it. The paper already gives you every hyperparameter. The only things the code adds are preprocessing edge cases and the literal 9→3 mapping dict, both recoverable. My earlier research couldn't retrieve this repo at all, so treat it as optional.

The discipline that matters more than the cloning: put all three in a reference/ directory that is outside your Python path and that you never import from. HiGrEC's claim is that structural inductive bias drives the gain. If TeleAI's prompt scaffolding leaks into your system, your gain is confounded with their prompt engineering, and that is the first thing a reviewer will attack. The single sanctioned exception is using a strong model to distil C5 span labels — and you disclose it.

I've written the prompts as a working document you'll paste from over several weeks.

Nineteen prompts across ten phases. Three things worth flagging about how I sequenced them:

Phase 1.2 comes before any modelling. You reimplement the official scorer and prove your version matches theirs with 2000+ randomized conformance tests. There's a specific property in there — that the score is invariant to which member of the reference set you predict — and if that property doesn't hold in the real scorer, C4's derivation is wrong and you need to know on day two, not in week four.

Phase 2 runs before you train anything. The Krippendorff alpha partition test needs no GPU and no annotation, and it's your most publishable result. I've built pre-registration into it as a separate committed file, because the test is only worth anything if you write down the prediction before seeing the number.

Every contribution has a control designed to kill it. C1 has a random-code control — if a randomly permuted but still-unique code performs as well as your semantic one, the mechanism is regularization, not structure. C5 has the shuffle control. C4 has the fixed-model-plus-majority-vote control and a nested-CV gap report. These are the ablations a reviewer would demand, and running them yourself is the difference between a course paper and a submission.

One planning risk to watch: if Phase 1.1 comes back saying the multi-reference labels aren't obtainable from what you can download, C3 and C4 as written are blocked. That's the single finding that would force a replan, which is why the data audit is prompt three rather than prompt ten.