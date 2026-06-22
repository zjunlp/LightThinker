EVALUATION_PROMPT = """As a rigorous Math Data Evaluator, your task is to verify the correctness of a 'Predicted Answer' against a 'Ground Truth Answer' given a specific 'Question', and extract a standardized final result.

Question: {question}
Predicted Answer: {predicted_answer}
Ground Truth Answer: {ground_truth_answer}

---
**Evaluation Criteria:**

1.  **Strict Adherence to Ground Truth (Do NOT Re-solve):**
    *   **Do NOT** attempt to solve the `Question` yourself. You must assume the `Ground Truth Answer` is the absolute correct logic.
    *   Compare the `Predicted Answer` strictly against the `Ground Truth Answer`.

2.  **Handling Corrupted Ground Truth:**
    *   If the `Ground Truth` appears to be corrupted, truncated, or contains a syntax error (e.g., `\\frac{1` instead of `\\frac{1}{2}`, or missing brackets), do NOT solve the math from scratch.
    *   Instead, inspect the `Question` (especially if it contains options) to **restore** the intended format of the Ground Truth.
    *   *Example:* If GT is `C`, but the Question has options `(C) 4`, treat the GT as `4`.
    *   *Example:* If GT is `\\frac{1`, and the Predicted Answer is `\\frac{1}{2}` which matches the context, accept the restoration.

3.  **Equivalence & Formatting:**
    *   Ignore units, currency symbols, and text unless strictly required (e.g., `$31.00` == `31`, `34 hours` == `34`, `14,000` == `14000`).
    *   Accept different valid notations (e.g., `0.5` == `\\frac{1}{2}`).
    *   If the Predicted Answer matches the Ground Truth (or the restored Ground Truth) mathematically, mark as **CORRECT**.
    *   If they do not match, mark as **INCORRECT**.

4.  **Standardized Answer Output:**
    *   You must extract the **Standardized Answer**.
    *   This should be the pure mathematical value (e.g., remove `$`, `units`, `\\text{}`, etc.).
    *   If Correct -> Use the cleaned `Predicted Answer`.
    *   If Incorrect -> Use the cleaned `Ground Truth Answer` (restored if necessary).

---
**Output Format:**

You must output exactly in the following format:

Thoughts: [Step 1: Check if GT is corrupted/truncated and restore if needed based on context. Step 2: Compare Predicted vs GT (equivalence check). Do not re-solve the math.]
Correctness: [TRUE or FALSE]
Standardized Answer: [The final clean mathematical string]
"""
