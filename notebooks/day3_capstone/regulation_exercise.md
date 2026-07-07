# Exercise: You're the regulator (5 minutes, in pairs)

You've seen where medical AI is deployed, why the hard part is institutional, and how it
breaks. Now you decide the rules.

**The question:** What must a medical AI system *prove* before it is allowed to touch a
patient, and how would you actually *check* it?

There is no answer key. Real regulators in the US, EU, and Asia disagree on exactly this.

## Candidate priorities (react to these, add your own)

| priority | the question it asks |
|----------|----------------------|
| **Safety** | does it actually work, and not harm? |
| **Fairness** | does it work for every group of patients? |
| **Transparency** | can a clinician see *why* it decided? |
| **Evidence** | tested where, on whom, how recently? |
| **Privacy** | is patient data protected? |
| **Monitoring** | is it watched *after* it ships? |
| **Accountability** | who is responsible when it's wrong? |

## Your task

1. **Pick your top 3** priorities. Why those?
2. For each, invent **one concrete way to check or enforce it** — a *tool*, not a wish.
3. Which of your three is the **hardest to actually enforce**, and why?

## How the real world enforces each (the reveal — compare after you've tried)

There's no single right answer, but here's one real tool per priority. Notice each is a *mechanism*
someone has to build, run, and pay for — that's why the hard part is institutional.

| priority | an example enforcement tool |
|----------|-----------------------------|
| **Safety** | a **clinical trial + regulatory clearance** (e.g. FDA 510(k)/De Novo) with a minimum performance bar it must clear *before* it can be sold. |
| **Fairness** | a **subgroup audit**: report accuracy/sensitivity *separately* by age, sex, and skin tone, and block approval if any group is much worse. |
| **Transparency** | a **model card** (documents the training data, intended use, and limits) plus a **saliency map** shown with each prediction so a clinician sees *where* the model looked. |
| **Evidence** | **external validation**: a peer-reviewed study at a hospital that did **not** build the model, on its *own* patients. |
| **Privacy** | **de-identification + access logs**, or **federated learning** (the model trains without patient data ever leaving the hospital). |
| **Monitoring** | **post-market surveillance**: a live dashboard watching accuracy for **drift**, mandatory incident reporting, and re-validation on a schedule or the approval **expires**. |
| **Accountability** | a **named responsible person/org**, an **audit trail** of every AI decision, and a **human override** (the AI advises, a person decides and signs off). |

## Talk about it

- Where did you and your partner disagree?
- The real world: the **US** approves device-by-device (FDA), the **EU** regulates by risk
  tier (the EU AI Act), and much of **Asia** moves fastest to deploy. None has fully solved
  generative/LLM tools in the clinic. Your list is a real, open policy debate.
