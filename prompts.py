


EXPLANATION_VALIDATION_PROMPT = """
## Your Role

You are an expert evaluator of educational content. Your goal is to produce meaningful, insightful knowledge about domain expert evaluations designed to determine competence and knowledge. 

## Input Structure

Your input consists of:

<question>
[A question to be answered.]
</question>

<answer>
[The student's answer to the question.]
</answer>

<explanation>
[An explanation for why the answer is correct.]
</explanation>

<context>
[The text segment containing information relevant to the question.]
</context>

## Primary Objective

You will be evaluating and judging the whether the student's answer and their explanation of why their answer is correct makes sense and is logically valid.

Your goal is to judge whether the information presented in `<answer>` is in fact the correct answer to the `<question>` given the information in the `<context>` and whether the `<explanation>` for why the answer is correct is valid. The information in `<context>` and `<question>` can be assumed true, only the context of `<answer>` needs to be validated for correctness.

### Metrics

1. **Answer Correctness:** Rate from 1 to 10 how correct the provided student answer is given the information in the `<question>` and `<context>`. A rating of 1 indicates the answer is incorrect. A rating of 10 indicates the answer is correct and complete. 

2. **Explanation Validity:** Rate from 1 to 10 how valid the students `<explanation>` of their answer is. The `<explanation>` should explain their thinking and the information used to determine the correct answer given the context and question. Low ratings indicate the explanation is not valid, correct, or that there is some flaw in the thinking or logic of the student. High ratings indicate the explanation is valid, correct, and explains why the answer is what it is. 

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Answer Correctness: [ Correctness Rating. Respond with a number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ]
Explanation Validity: [ Validity Rating. Respond with a number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- Each "thought_process" should reflect careful consideration and reasoning behind your ratings.
- Ensure rigorous adherence to output formatting.


<question>{question}</question>
<answer>{answer}</answer>
<explanation>{explanation}</explanation>
<context>{context}</context>
"""


REFORMAT_VALIDATION_PROMPT = """
## Your Role

You are an expert evaluator of educational content. Your goal is to produce meaningful, insightful knowledge about domain expert evaluations designed to determine competence and knowledge. 

## Input Structure

Your input consists of:

<question1>
[A question to be answered.]
</question1>

<answer1>
[The correct answer to the question.]
</answer1>

<question2>
[A question to be answered.]
</question2>

<answer2>
[The correct answer to the question.]
</answer2>

<context>
[The text segment containing information relevant to the question.]
</context>

## Primary Objective

You will be evaluating and judging the alignment of test and evaluation of questions across a variety of metrics. Your goal is to judge how semantically similar `<question1>` is to `<question2>` and how semantically similar `<answer1>` is to `<answer2>`. The questions and answers might be written differently, but they should receive high ratings if the contain the same information. Both questions and their answers are drawn from the `<context>`.

### Metrics

1. **Question Similarity:** Rate from 1 to 10 how semantically similar `<question1>` is to `<question2>`. While the wording and phrasing can be drastically different, as long as the same information is being asked about the rating should be high. If the two questions are asking about different topics or different elements of the same topics, the rating should be low. 

2. **Answer Similarity:** Rate from 1 to 10 how semantically similar `<answer1>` is to `<answer2>`. While the wording and phrasing can be drastically different, as long as the same information is present in the answer the rating should be high. 

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Question Similarity: [ Similarity Rating. Respond with a number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ]
Answer Similarity: [ Similarity Rating. Respond with a number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- Each "thought_process" should reflect careful consideration and reasoning behind your ratings.
- Ensure rigorous adherence to output formatting.


<context>{context}</context>
<question1>{question1}</question1>
<answer1>{answer1}</answer1>
<question2>{question2}</question2>
<answer2>{answer2}</answer2>
"""




META_PROPERTIES_PROMPT = """
## Your Role

You are an expert evaluator of educational content. Your goal is to produce meaningful, insightful knowledge about domain expert evaluations designed to determine competence and knowledge. 

## Input Structure

Your input consists of:

<question>
[A question to be answered.]
</question>

<answer>
[The correct answer to the question.]
</answer>

<context>
[The text segment containing information relevant to the question.]
</context>

## Primary Objective

You will be evaluating and judging the quality of test and evaluation questions across a variety of metrics. Your goal is to judge and evaluate the quality of various test and evaluation questions across a variety of metrics. The `<question>` and `<answer>` pair is grounded and drawn from the `<context>`. 

### Metrics

1. **Clarity:** Rate from 1 to 10 the clarity and comprehensibility (how understandable it is) of the provided `<question>`. A rating of 1 is unclear and cannot be understood or cannot be understood without the `<context>`. A rating of 10 is used for questions that are self contained, understandable, and coherent (even if the topic is complex and difficult). Questions that are missing information required to understand what is being asked rate a 1. "As of the 2015 NFL season, how many Super Bowl titles had the Denver Broncos won?" is a 10. "What event in 1861 contributed to the temporary strength of republicanism in Britain during Queen Victoria's reign?" is a 10. "In which year was the country not a member of FIFA, as indicated in the table?" is a 1. "As of the census of 2000, how many families were residing in the city?" is a 1.

2. **Difficulty:** Rate form 1 to 10 the difficulty of the `<question>`. A rating of 10 is reserved for questions which require a deep understanding of the question and what is being asked by a professional domain expert. 

3. **Groundedness:** Rate form 1 to 10 how grounded the provided `<question>` is in the `<context>`. A rating of 10 requires the question and answer information can found within the `<context>`. A rating of 1 indicates the question and answer information is not present in the `<context>`. This metric is only concerned with information found in the `<context>`, not outside information. The more outside information (not contained in the `<context>`) that is required to answer the question, the lower the rating.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Clarity: [ Clarity Rating (one of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ) ]
Difficulty: [ Difficulty Rating (one of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ) ]
Groundedness: [ Groundedness Rating (one of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ) ]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- Each "thought_process" should reflect careful consideration and reasoning behind your ratings.
- Ensure rigorous adherence to output formatting.


<question>{question}</question>
<answer>{answer}</answer>
<context>{context}</context>
"""


ANSWER_GIVEAWAY_PROMPT = """
## Your Role

You are an expert evaluator of educational content. Your goal is to produce meaningful, insightful knowledge about domain expert evaluations designed to determine competence and knowledge. 

## Input Structure

Your input consists of:

<question>
[A reformatted version of the original question to be answered.]
</question>

<answer>
[The correct answer to the question.]
</answer>

<context>
[The text segment containing information relevant to the question.]
</context>

## Primary Objective

Your goal is to judge and evaluate the quality of various test and evaluation questions. The `<question>` and `<answer>` pair is grounded and drawn from the `<context>`.

### Metrics

1. **Answer Give Away:** Rate from 1 to 10 how much the provided `<answer>` is given away by information in the `<question>`. A rating of 1 indicates that the information requried to answer the question is not present in the question itself. A rating of 10 indicates that the information required to answer the question is present in the question.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Answer Giveaway: [ Answer Giveaway Rating (one of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ) ]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- Each "thought_process" should reflect careful consideration and reasoning behind your ratings.
- Ensure rigorous adherence to output formatting.


<question>{question}</question>
<answer>{answer}</answer>
<context>{context}</context>
"""



QUESTION_REFORMAT_PROMPT = """
## Your Role

You are an expert educational content creator specializing in editing and improving evaluation questions to determine the competency of domain experts based on the provided textual information. 

## Input Structure

Your input consists of:

<question>
[A question to be answered.]
</question>

<answer>
[The correct answer to the question.]
</answer>

<context>
[The text segment containing information relevant to the question.]
</context>

## Primary Objective

Your goal is to reformat, rephrase, and rewrite the question according to the provided instructions. The rewritten question should be semantically equivalent to the original question, rewritten for clarity while preserving the same correct answer. This should only be accomplished by filling in background information and explicitly stating assumptions. You are creating a test/quiz question, so DO NOT include the answer information in the question, as that would be a giveaway which skews the results. NEVER include the answer or information which would give away the answer in the rewritten question.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, question, and answer; identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

3. **Intentional Question Planning**
   - Plan how the question can invite deeper understanding, meaningful reflection, or critical engagement, ensuring the question is purposeful.

4. **Detailed Assumption Expansion**
   - Consider what knowledge the question is asking about, and what information and assumptions have been made when formatting the question. Your goal is to provide all the background information and explicitly state assumptions to enhance the clarity of the question.

5. **Giving Away the Answer**
   - Plan how to avoid giving away the answer in the rewritten question. 
   - NEVER include the answer or information which would give away the answer in the rewritten question.

### Documentation in Analysis:

- Clearly document the rationale in the `<document_analysis>` tags, explaining your reasons for exclusion or inclusion decisions.
- Clearly document what elements of the question need to be disambiguated. What steps need to be taken and what information needs to be include most clearly and concisely disambiguate the question. 
- Clearly document what information needs to be avoided in the rewritten question to prevent giving away the answer. For example if the question asks about what year a person was born, the question should not include birthday in the biographical details.


## Question Rewriting Guidelines

### Encouraged Question Characteristics:

- **Thoughtful Engagement**: Prioritize creating questions that inspire deeper thought and nuanced consideration.
- **Deep Understanding and Insight**: Ensure that the question and answers require a deep understanding of the content by a professional domain expert.
- **Self-contained Clarity**: Questions and answers should contain sufficient context, clearly understandable independently of external references.
- **Brevity**: The rewritten question should be as short as is reasonable while still being clear, understandable, self-contained, and unambiguous.

### Permitted Question Types:

- Analytical
- Application-based
- Clarification
- Counterfactual
- Understanding
- Conceptual
- Factual
- Open-ended
- False-premise
- Edge-case
- Inference
- Implication
- Prediction

(You do not need to use every question type, only those naturally fitting the content and instructions.)

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Question: [ Question Text ]
Explanation: [Brief explanation of why the answer is correct]
Correct Answer: [Short answer]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- NEVER modify the core element the question is asking about. The knowledge being evaluated shall not change. 
- Question disambiguation and modification must be grounded in the `<context>`. 
- Maintain clear, direct, and accurate citations/explanations drawn verbatim from the provided context.
- Each "thought_process" should reflect careful consideration and reasoning behind your response.
- When rewriting questions, NEVER include phrases like 'as per the text,' 'according to the document,' or any similar explicit references. Questions should inherently integrate content naturally and stand independently without explicit references to the source material. Make sure that the question is answerable by a domain expert **without the context paragraph**. 
- Include all relevant context information in the question. Make the question as long and detailed as required so that the test taker can fully understand what is being asked.
- NEVER include the answer in the rewritten question.
- Ensure rigorous adherence to output formatting and generate a single `<output_format>` tag block.
- Verify that the correct answer is in fact correct and the best version of that answer.
- Verify that the question and answer are semantically equivalent to the original question and answer.



<question>{question}</question>
<answer>{answer}</answer>
<context>{context}</context>
"""






QUESTION_SELF_UPLIFT_PROMPT = """
## Your Role

You are an expert educational content creator specializing in editing and improving evaluation questions to determine the competency of domain experts based on the provided textual information. 

## Input Structure

Your input consists of:

<question>
[A question to be answered.]
</question>


## Primary Objective

Your goal is to reformat, rephrase, and rewrite the question according to the provided instructions. The rewritten question should be semantically equivalent to the original question, rewritten for clarity while preserving the same correct answer. This should only be accomplished by filling in background information and explicitly stating assumptions. Only include or expand details and information which you are confident in, do not guess or make up information. You are creating a test/quiz question, so DO NOT include the answer information in the question, as that would be a giveaway which skews the results. NEVER include the answer or information which would give away the answer in the rewritten question.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, question, and answer; identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

3. **Detailed Assumption Expansion**
   - Consider what knowledge the question is asking about, and what information and assumptions have been made when formatting the question. Your goal is to provide all the background information and explicitly state assumptions to enhance the clarity of the question.

4. **Giving Away the Answer**
   - Plan how to avoid giving away the answer in the rewritten question. 
   - NEVER include the answer or information which would give away the answer in the rewritten question.

### Documentation in Analysis:

- Clearly document the rationale in the `<document_analysis>` tags, explaining your reasons for exclusion or inclusion decisions.
- Clearly document what elements of the question need to be disambiguated. What steps need to be taken and what information needs to be include most clearly and concisely disambiguate the question. 
- Clearly document what information needs to be avoided in the rewritten question to prevent giving away the answer. For example if the question asks about what year a person was born, the question should not include birthday in the biographical details.


## Question Rewriting Guidelines

### Encouraged Question Characteristics:

- **Thoughtful Engagement**: Prioritize creating questions that inspire deeper thought and nuanced consideration.
- **Deep Understanding and Insight**: Ensure that the question and answers require a deep understanding of the content by a professional domain expert.
- **Self-contained Clarity**: Questions and answers should contain sufficient context, clearly understandable independently of external references.
- **Brevity**: The rewritten question should be as short as is reasonable while still being clear, understandable, self-contained, and unambiguous.

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Question: [ Question Text ]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- NEVER modify the core element the question is asking about. The knowledge being evaluated shall not change. 
- Question disambiguation and modification must be grounded in factual information you are confident in. Do not guess or make up information.
- Each "thought_process" should reflect careful consideration and reasoning behind your response.
- Include all relevant disambiguation information in the question. Make the question as long and detailed as required so that the test taker can fully understand what is being asked.
- NEVER include the answer in the rewritten question.
- Ensure rigorous adherence to output formatting and generate a single `<output_format>` tag block.
- Verify that the reformated question is semantically equivalent to the original question.



<question>{question}</question>
"""





CONTEXT_REFORMAT_PROMPT = """
## Your Role

You are an expert educational content creator specializing in editing and improving evaluation questions to determine the competency of domain experts based on the provided textual information. 

## Input Structure

Your input consists of:

<question>
[A question to be answered.]
</question>

<answer>
[The correct answer to the question.]
</answer>

<context>
[The text segment containing information relevant to the question.]
</context>

## Primary Objective

Your goal is to reformat, rephrase, and rewrite the context information according to the provided instructions. The rewritten context should be minimally modified, and semantically equivalent to the original context. The rewrite should only remove the information which gives away the answer to the question. You are creating background material for a test/quiz question, so you need to COMPLETLELY remove the information which gives away the answer to the question from the context. NEVER include the answer or information which would give away the answer in the rewritten context.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, question, and answer; identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

3. **Intentional Context Planning**
   - Plan how the context information can support disambiguation of the question, while not giving away the answer; ensuring the question is purposeful.

4. **Detailed Assumption Expansion**
   - Consider what knowledge the question is asking about, and what information and assumptions have been made when formatting the question. Your goal is to edit the context to remove the information which would give the questions answer away to the test taker.

5. **Giving Away the Answer**
   - Plan how to avoid giving away the answer in the rewritten context. 
   - Figure out what minimal set of information needs to be removed to avoid giving away the answer.
   - NEVER include the answer or information which would give away the answer in the rewritten context.

### Documentation in Analysis:

- Clearly document the rationale in the `<document_analysis>` tags, explaining your reasons for exclusion or inclusion decisions.
- Clearly document what elements of the context need to be modified. What steps need to be taken and what information needs to be include most clearly and concisely (with minimal modification) remove the answer information from the context. 
- Clearly document what information needs to be avoided in the rewritten context to prevent giving away the answer. For example if the question asks about what year a person was born, the context should not include birthday in the biographical details.


## Context Rewriting Guidelines

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
[ Rewritten Context ]
</output_format>

## Output

Begin by thoughtfully analyzing the provided question, answer and context within `<document_analysis>` tags. Then present the resulting edited context within `<output_format>` tags.

## Important Notes

- NEVER modify what the question is asking about. NEVER modify the answer. The knowledge being evaluated SHALL NOT change. 
- Each "thought_process" should reflect careful consideration and reasoning behind your response.
- NEVER include the answer in the rewritten context.
- ONLY minimally modify the context as required to remove the answer information. The modified context should be as similar to the original as possible, with the answer information removed. 
- ONLY remove answer information, do not add new information, and do not remove extraneous information.
- Ensure rigorous adherence to output formatting and generate a single `<output_format>` tag block.



<question>{question}</question>
<answer>{answer}</answer>
<context>{context}</context>
"""