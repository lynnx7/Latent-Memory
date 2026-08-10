from dataclasses import dataclass
import re
from typing import Optional

from inference_utils.reflector_tool_notes import system_prompt as REFLECTOR_TOOL_SYSTEM_PROMPT

SYSTEM_PROMPT = "You are a helpful assistant."
DELIBERATION_SYSTEM_PROMPT = REFLECTOR_TOOL_SYSTEM_PROMPT

PLANNER_SLOT = "<<LATENT_PLANNER_SLOT>>"
REFINED_SLOT = "<<LATENT_REFINED_SLOT>>"
FEEDBACK_SLOT = "<<LATENT_FEEDBACK_SLOT>>"

HIE_MATH_EXPERT_SLOT = "<<HIE_MATH_EXPERT_SLOT>>"
HIE_CODE_EXPERT_SLOT = "<<HIE_CODE_EXPERT_SLOT>>"
HIE_SCIENCE_EXPERT_SLOT = "<<HIE_SCIENCE_EXPERT_SLOT>>"
HIE_FEEDBACK_SLOT = "<<HIE_FEEDBACK_SLOT>>"
DISTILL_EXPERT_SLOT = "<<DISTILL_EXPERT_SLOT>>"
DISTILL_FEEDBACK_SLOT = "<<DISTILL_FEEDBACK_SLOT>>"
DELIBERATION_REFLECTOR_SLOT = "<<DELIBERATION_REFLECTOR_SLOT>>"
DELIBERATION_FEEDBACK_SLOT = "<<DELIBERATION_FEEDBACK_SLOT>>"

SYN_SLOT = "<<LATENT_SYN_SLOT>>"
OLD_SYN_SLOT = "<<OLD_LATENT_SYN_SLOT>>"

@dataclass(frozen=True)
class MASPromptBundle:
    planner_user: str
    refiner_user: str
    solver_user: str

# 新增 
# def build_synthesize_prompt_with_slot(question: str) -> str:
#     return (
#         f"You are a Memory Synthesis Agent in a multi-agent problem-solving process.\n"
#         f"Below is the problem statement:\n{question}\n\n"
#         f"The collective latent reasoning process from all agents is encoded here:\n"
#         f"{SYNTHESIZE_SLOT}\n\n"
#         f'''You are responsible for maintaining, refining, and optimizing the memory, which serves as a compact yet evolving repository
#             of problem-solving strategies, reusable code snippets, and meta-reasoning techniques. Your goal is to enhance the model’s long-term
#             performance by continuously updating the memory with high-value insights while filtering out redundant or trivial information.
#             - The memory should include quick, accurate, reliable, and practical solutions to a range of technical and creative challenges.
#             - After seeing each input, you should improve the content of the memory, synthesizing lessons, insights, tricks, and errors learned from
#             past problems and adapting to new challenges.'''
#     )

#使用记忆
# def build_code_planner_prompt_with_mem(question: str, task_type: str, fn_name: Optional[str] = None) -> str:
#     interface = build_code_interface_prompt(task_type, fn_name=fn_name)
#     return (
#         "You are a planner agent in a multi-agent coding system.\n"
#         f"{interface}\n"
#         "The programming problem is:\n"
#         f"{question}\n"
#         "Memory from previous tasks:\n"
#         f"{SYN_SLOT}\n"
#         "Use the memory as a soft correction signal to improve the plan.\n"
#         "If there is any conflict, prioritize the problem constraints.\n"
#         "Provide a clear step-by-step plan (within 3-6 steps) to solve the problem.\n"
#         "Do not write code.\n"
#         "Your response should be in the format of:\n"
#         "Step 1: ...\n"
#         "...\n"
#         "Step n: ..."
        
#     )
# def build_code_planner_prompt_with_mem(question: str, task_type: str, fn_name: Optional[str] = None) -> str:
#     interface = build_code_interface_prompt(task_type, fn_name=fn_name)
#     return (
#         "You are a planner agent in a recursive multi-agent coding system.\n"
#         "This is a later recursive round.\n"
#         f"{interface}\n"
#         "The programming problem is:\n"
#         f"{question}\n"
#         "Feedback signal from the previous solver round:\n"
#         f"{SYN_SLOT}\n"
#         "Use the feedback as a soft correction signal to improve the plan.\n"
#         "If there is any conflict, prioritize the problem constraints.\n"
#         "Provide a clear step-by-step plan (within 3-6 steps) to solve the problem.\n"
#         "Do not write code.\n"
#         "Your response should be in the format of:\n"
#         "Step 1: ...\n"
#         "...\n"
#         "Step n: ..."
        
#     )

def build_code_planner_prompt_with_text_mem(question: str, past_q:str, past_plan:str, task_type: str, fn_name: Optional[str] = None) -> str:
    interface = build_code_interface_prompt(task_type, fn_name=fn_name)
    return (
        "You are a planner agent in a recursive multi-agent coding system.\n"
        f"{interface}\n"
        "Relevant Previous Example:\n"
        "Problem:\n"
        f"{past_q}\n"
        "\nPlan:\n"
        f"{past_plan}\n"
        "\nThe current programming problem is:\n"
        f"{question}\n"
        "Provide a clear step-by-step plan (within 3-6 steps) to solve the problem.\n"
        "Do not write code.\n"
        "Your response should be in the format of:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
        
    )

#备用
# def build_synthesize_prompt_with_slot(question: str) -> str:
#     return (
#         f"You are a Memory Synthesis Agent in a multi-agent problem-solving process.\n"
#         f"Below is the problem statement:\n{question}\n\n"
#         f"The collective latent reasoning process from Refiner and Solver agents is encoded here:\n"
#         f"{SYN_SLOT}\n\n"
#         f"Based on this, summarize the key insights, and general strategies."
#     )

# def build_synthesize_prompt_with_slot(question: str) -> str:
#     return (
#         f"You are a Memory Synthesis Agent in a multi-agent problem-solving process.\n"
#         f"Below is the problem statement:\n{question}\n\n"
#         f"Summarize and analyse the difficulty and trait of this problem."
#     )

def build_synthesize_prompt_with_slot(question: str) -> str:
    return (
        f"You are a Memory Generator in a multi-agent system.\n"
        f"Below is the problem statement:\n{question}\n\n"
        f"The collective latent reasoning process from previous agents is encoded here:\n"
        f"{SYN_SLOT}\n\n"
        f"Summarize the updated key insights, and general strategies."
        f"Do not write code.\n"
        f"No need to give solution.\n"
        f"Your response should be in the format of:\n"
        f"<mem>"
        f"..."
        f"</mem>"

        
    )

#之后更新记忆
def build_updated_synthesize_prompt_with_slot(question: str) -> str:
    return (
        f"You are a Memory Generator in a multi-agent system.\n"
        f"Below is the problem statement:\n{question}\n\n"
        f"The collective latent reasoning process from previous agents is encoded here:\n"
        f"{SYN_SLOT}\n\n"
        f"Based on previous memory {OLD_SYN_SLOT}, summarize the updated key insights, and general strategies."
        f"Do not write code.\n"
        f"No need to give solution.\n"
    )
    

def get_system_prompt(mas_design: str = "chain", mas_role: Optional[str] = None) -> str:
    role_name = str(mas_role or "").strip().lower()
    design_name = str(mas_design or "chain").strip().lower()
    if design_name == "deliberation" or role_name in {"deliberation_reflector", "deliberation_toolcaller"}:
        return DELIBERATION_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def _normalize_code_task_type(task_type: str) -> str:
    task_type = str(task_type or "").strip().lower()
    if task_type in {"function", "functional"}:
        return "function"
    return "complete"


def build_code_interface_prompt(task_type: str, fn_name: Optional[str] = None) -> str:
    mode = _normalize_code_task_type(task_type)
    if mode == "function":
        if fn_name:
            return f"Implement and return the function `{fn_name}` only."
        return "Implement and return the required function only."
    return "Write a complete program that reads from stdin and prints to stdout."


def build_hie_task_context(
    question: str,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> str:
    mas_task = str(mas_task or "math").strip().lower()
    if mas_task == "code":
        interface = build_code_interface_prompt(task_type, fn_name=fn_name)
        return (
            f"{interface}\n"
            "The programming problem is:\n"
            f"{question}\n"
        )
    if mas_task == "choice":
        question = str(question).rstrip()
        if "choose the correct option" not in question.lower():
            question = (
                f"{question}\n\n"
                "Choose the correct option (A/B/C/D)."
            )
        return (
            "The question is:\n"
            "Question:\n"
            f"{question}\n"
        )
    return (
        "The question is:\n"
        "Question:\n"
        f"{question}\n"
    )


def _hie_final_instruction(question: str, mas_task: str = "math") -> str:
    mas_task = str(mas_task or "math").strip().lower()
    if mas_task == "code":
        return (
            "Solve the problem and put the final code inside one markdown code block, "
            "for example ```python\\n<your solution code>\\n```."
        )
    if mas_task == "choice":
        return "Solve the question and put the final choice inside \\boxed{}, for example \\boxed{A}."
    is_choice_question = bool(re.search(r"(?mi)^\s*[A-D]\s*[\.\):\-]\s+", question))
    if is_choice_question:
        return "Solve the question and put the final choice inside \\boxed{}, for example \\boxed{A}."
    return "Solve the question and put the final answer inside \\boxed{}, for example \\boxed{1}."


def build_hie_expert_prompt(
    question: str,
    hie_role: str,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> str:
    role_name = str(hie_role or "").strip().lower()
    role_labels = {
        "hie_math_expert": "math expert",
        "hie_code_expert": "code expert",
        "hie_science_expert": "science expert",
    }
    if role_name not in role_labels:
        raise ValueError(f"Unsupported hie expert role: {hie_role}")

    task_context = build_hie_task_context(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )
    return (
        f"You are the {role_labels[role_name]} in a multi-agent system.\n"
        f"{task_context}"
    )


def build_hie_expert_prompt_with_feedback_slot(
    question: str,
    hie_role: str,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> str:
    role_name = str(hie_role or "").strip().lower()
    role_labels = {
        "hie_math_expert": "math expert",
        "hie_code_expert": "code expert",
        "hie_science_expert": "science expert",
    }
    if role_name not in role_labels:
        raise ValueError(f"Unsupported hie expert role: {hie_role}")

    task_context = build_hie_task_context(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )
    return (
        f"You are the {role_labels[role_name]} in a multi-agent system.\n"
        "Feedback signal from the previous summarizer round:\n"
        f"{HIE_FEEDBACK_SLOT}\n"
        "Use the feedback as a soft correction signal, and provide a concise updated expert answer for the summarizer.\n"
        f"{task_context}"
    )


def build_hie_summarizer_prompt_with_slots(
    question: str,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> str:
    task_context = build_hie_task_context(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )
    return (
        "You are the summarizer agent in a multi-agent system.\n"
        "Math expert signal:\n"
        f"{HIE_MATH_EXPERT_SLOT}\n"
        "Code expert signal:\n"
        f"{HIE_CODE_EXPERT_SLOT}\n"
        "Science expert signal:\n"
        f"{HIE_SCIENCE_EXPERT_SLOT}\n"
        f"---\n"
        "You may reference the three expert information.\n"
        f"Please reason step by step and solve the problem below:\n"
        f"{task_context}"
        f"{_hie_final_instruction(question, mas_task=mas_task)}"
    )


def build_hie_summarizer_prompt(
    question: str,
    math_expert_output: str,
    code_expert_output: str,
    science_expert_output: str,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> str:
    task_context = build_hie_task_context(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )
    return (
        "You are the summarizer agent in a multi-agent system.\n"
        "Math expert signal:\n"
        f"{math_expert_output}\n"
        "Code expert signal:\n"
        f"{code_expert_output}\n"
        "Science expert signal:\n"
        f"{science_expert_output}\n"
        f"---\n"
        "You may reference the three expert information.\n"
        f"Please reason step by step and solve the problem below:\n"
        f"{task_context}"
        f"{_hie_final_instruction(question, mas_task=mas_task)}"
    )


def build_distill_expert_prompt(
    question: str,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> str:
    task_context = build_hie_task_context(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )
    return (
        "You are the expert agent in a multi-agent system.\n"
        f"{task_context}"
        "Provide a concise, execution-ready plan that the learner can follow.\n"
        "Do not provide the final answer.\n"
        "Your response should be in the format:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )


def build_distill_expert_prompt_with_feedback_slot(
    question: str,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> str:
    task_context = build_hie_task_context(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )
    return (
        "You are the expert agent in a multi-agent system.\n"
        "Feedback signal from the previous round:\n"
        f"{DISTILL_FEEDBACK_SLOT}\n"
        "Use the feedback as a soft correction signal.\n"
        f"{task_context}"
        "Provide a concise, execution-ready updated plan.\n"
        "Do not provide the final answer.\n"
        "Your response should be in the format:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )


def build_distill_learner_prompt(
    question: str,
    expert_plan: str,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> str:
    task_context = build_hie_task_context(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )
    return (
        "You are the learner executor in a multi-agent system.\n"
        "Expert plan:\n"
        f"{expert_plan}\n"
        "Use the expert plan as guidance, but prioritize the task constraints.\n"
        f"---\n"
        f"{task_context}"
        f"{_hie_final_instruction(question, mas_task=mas_task)}"
    )


def build_distill_learner_prompt_with_slot(
    question: str,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> str:
    return build_distill_learner_prompt(
        question=question,
        expert_plan=DISTILL_EXPERT_SLOT,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )


def build_deliberation_reflector_prompt(
    question: str,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> str:
    task_context = build_hie_task_context(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )
    return (
        "You are the reflector agent in a deliberation style multi-agent system.\n"
        "You may use tools following the system instructions when helpful.\n"
        f"{task_context}"
    )


def build_deliberation_reflector_prompt_with_feedback_slot(
    question: str,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> str:
    task_context = build_hie_task_context(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )
    return (
        "You are the reflector agent in a deliberation style multi-agent system.\n"
        "Toolcaller feedback signal from the previous round:\n"
        f"{DELIBERATION_FEEDBACK_SLOT}\n"
        "Use the feedback as a soft guidance.\n"
        "You may use tools following the system instructions when helpful.\n"
        f"{task_context}"
    )


def build_deliberation_toolcaller_prompt(
    question: str,
    reflector_signal: str,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> str:
    task_context = build_hie_task_context(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )
    return (
        "You are the toolcaller agent in a deliberation style multi-agent system.\n"
        "Reflector signal:\n"
        f"{reflector_signal}\n"
        "Use the reflector signal as soft guidance.\n"
        "You may use tools following the system instructions when helpful.\n"
        f"---\n"
        f"{task_context}"
        f"{_hie_final_instruction(question, mas_task=mas_task)}"
    )


def build_deliberation_toolcaller_prompt_with_slot(
    question: str,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> str:
    return build_deliberation_toolcaller_prompt(
        question=question,
        reflector_signal=DELIBERATION_REFLECTOR_SLOT,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )


def build_code_planner_prompt(question: str, task_type: str, fn_name: Optional[str] = None) -> str:
    interface = build_code_interface_prompt(task_type, fn_name=fn_name)
    return (
        "You are a planner agent in a multi-agent coding system.\n"
        f"{interface}\n"
        "The programming problem is:\n"
        f"{question}\n"
        "Provide a clear step-by-step plan (within 3-6 steps) to solve the problem.\n"
        "Do not write code.\n"
        "Your response should be in the format of:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )


def build_code_refiner_prompt(
    question: str,
    planner_output: str,
    task_type: str,
    fn_name: Optional[str] = None,
) -> str:
    interface = build_code_interface_prompt(task_type, fn_name=fn_name)

    # return (
    #     f"What does this say: {planner_output}?"
    # )
    
    return (
        "You are a refiner agent in a multi-agent coding system.\n"
        f"{interface}\n"
        "\n---\nThe programming problem is:\n"
        f"{question}\n"
        "The initial plan from the planner:\n"
        "Initial Plan:\n"
        f"{planner_output}\n"
        "Refine the plan into a clearer and stronger step-by-step plan (within 3-6 steps).\n"
        "Do not write code.\n"
        "Your response should be in the format of:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )


def build_code_solver_prompt(
    question: str,
    refined_plan: str,
    task_type: str,
    args=None,
    fn_name: Optional[str] = None,
    past_q: str = None,
    past_ans: str = None,
) -> str:
    interface = build_code_interface_prompt(task_type, fn_name=fn_name)
    final_instruction = (
        "Solve the problem and put the final code inside one markdown code block, "
        "for example ```python\\n<your solution code>\\n```."
    )

    if args is not None and int(getattr(args, "solver_pre_question", 0)) == 1:
        if past_q is None:
            return (
                "You are a solver agent in a multi-agent coding system.\n"
                f"{interface}\n"
                
                "Relevant Previous Example:\n"
                "Problem:\n"
                f"{past_q}\n"
                "\nSolution:\n"
                f"{past_ans}\n"
                "\n---\nThe current programming problem is:\n"
                f"{question}\n"
                "Here is the refined plan:\n"
                "Refined Plan:\n"
                f"{refined_plan}\n"
                f"{final_instruction}"
            )
        else:
            return (
                "You are a solver agent in a multi-agent coding system.\n"
                f"{interface}\n"
                "\n---\nThe programming problem is:\n"
                f"{question}\n"
                "\n---\nThe programming problem is:\n"
                f"{mem}\n"
                
                "Here is the refined plan:\n"
                "Refined Plan:\n"
                f"{refined_plan}\n"
                f"{final_instruction}"
            )

    return (
        "You are a solver agent in a multi-agent coding system.\n"
        f"{interface}\n"
        "Here is the refined plan:\n"
        "Refined Plan:\n"
        f"{refined_plan}\n"
        "\n---\nThe programming problem is:\n"
        f"{question}\n"
        f"{final_instruction}"
    )


def build_code_planner_prompt_with_feedback_slot(
    question: str,
    task_type: str,
    fn_name: Optional[str] = None,
) -> str:
    interface = build_code_interface_prompt(task_type, fn_name=fn_name)
    return (
        "You are a planner agent in a recursive multi-agent coding system.\n"
        "This is a later recursive round.\n"
        f"{interface}\n"
        "The programming problem is:\n"
        f"{question}\n"
        "Feedback signal from the previous solver round:\n"
        f"{FEEDBACK_SLOT}\n"
        "Use the feedback as a soft correction signal to improve the plan.\n"
        "If there is any conflict, prioritize the problem constraints.\n"
        "Provide a clear step-by-step plan (within 3-6 steps) to solve the problem.\n"
        "Do not write code.\n"
        "Your response should be in the format of:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )


def build_code_refiner_prompt_with_slot(
    question: str,
    task_type: str,
    fn_name: Optional[str] = None,
) -> str:
    return build_code_refiner_prompt(
        question=question,
        planner_output=PLANNER_SLOT,
        task_type=task_type,
        fn_name=fn_name,
    )


def build_code_solver_prompt_with_slots(
    question: str,
    task_type: str,
    args=None,
    mas_shape: str = "chain",
    fn_name: Optional[str] = None,
    past_q: str = None,
    past_ans: str = None,
) -> str:
    if mas_shape == "chain":
        return build_code_solver_prompt(
            question=question,
            refined_plan=REFINED_SLOT,
            task_type=task_type,
            args=args,
            fn_name=fn_name,
            past_q=past_q,
            past_ans=past_ans,
        )
    raise ValueError(f"Unsupported mas_shape: {mas_shape}")


def build_math_planner_prompt(question: str) -> str:
    return (
        "You are a planner agent in a multi-agent system.\n"
        "Give a plan for the question below.\n"
        "Question:\n"
        f"{question}\n"
        "Your response should be in the format of:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )


def build_math_planner_prompt_with_feedback_slot(question: str) -> str:
    return (
        "You are a planner agent in a recursive multi-agent system.\n"
        "This is round 2.\n"
        "Question:\n"
        f"{question}\n"
        "Feedback signal from the previous solver round:\n"
        f"{FEEDBACK_SLOT}\n"
        "Use the feedback as a soft correction signal to improve the plan.\n"
        "If there is any conflict, prioritize the question constraints.\n"
        "Output only a concise plan in the format:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )


def build_math_refiner_prompt(question: str, planner_output: str) -> str:
    return (
        "You are a refiner agent in a multi-agent system.\n"
        "The question is:\n"
        "Question:\n"
        f"{question}\n"
        "The initial plan from the planner:\n"
        "Initial Plan:\n"
        f"{planner_output}\n"
        "You should refine the initial plan and respond with pure plan only in the format of:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ..."
    )


def build_math_solver_prompt(
    question: str,
    refined_plan: str,
    args=None,
) -> str:
    is_choice_question = bool(
        re.search(r"(?mi)^\s*[A-D]\s*[\.\):\-]\s+", question)
    )
    choice_old_prompt_mode = (
        int(getattr(args, "choice_old_prompt", 0))
        if (is_choice_question and args is not None)
        else 0
    )
    use_choice_old_prompt = choice_old_prompt_mode in (1, 2, 3)
    if choice_old_prompt_mode == 1:
        final_instruction = (
            "Final Choice: put only the option letter in \\boxed{}, e.g., \\boxed{A}.\n\n"
            "Solve the question given information and put the final answer inside \\boxed{}, for example \\boxed{1}."
        )
    elif choice_old_prompt_mode == 2:
        final_instruction = (
            "Final Choice: put only the option letter in \\boxed{}, e.g., \\boxed{A}."
        )
    elif choice_old_prompt_mode == 3:
        final_instruction = (
            "Final Choice: put only the option letter in \\boxed{}."
        )
    else:
        final_instruction = (
            "Solve the question and put the final choice inside \\boxed{}, for example \\boxed{A}."
            if is_choice_question
            else "Solve the question given information and put the final answer inside \\boxed{}, for example \\boxed{1}."
        )
    if args is not None and args.solver_pre_question == 1:
        return (
            "You are a solver agent in a multi-agent system.\n"
            "The question is:\n"
            "Question:\n"
            f"{question}\n"
            "Here is the refined plan:\n"
            "Refined Plan:\n"
            f"{refined_plan}\n"
            f"{final_instruction}"
        )
    question_to_final_sep = "\n" if use_choice_old_prompt else "\n\n"
    return (
        "You are a solver agent in a multi-agent system.\n"
        "Here is the refined plan:\n"
        "Refined Plan:\n"
        f"{refined_plan}\n"
        "The question is:\n"
        "Question:\n"
        f"{question}{question_to_final_sep}"
        f"{final_instruction}"
    )


def build_math_refiner_prompt_with_slot(question: str) -> str:
    return build_math_refiner_prompt(question, PLANNER_SLOT)


def build_math_solver_prompt_with_slots(
    question: str,
    args=None,
    mas_shape: str = "chain",
) -> str:
    if mas_shape == "chain":
        return build_math_solver_prompt(question, REFINED_SLOT, args)
    raise ValueError(f"Unsupported mas_shape: {mas_shape}")


def build_math_prompt_bundle(
    question: str,
    planner_output: str,
    refined_output: str,
    args=None,
    mas_shape: str = "chain",
) -> MASPromptBundle:
    return MASPromptBundle(
        planner_user=build_math_planner_prompt(question),
        refiner_user=build_math_refiner_prompt(question, planner_output),
        solver_user=build_math_solver_prompt_with_slots(
            question,
            args=args,
            mas_shape=mas_shape,
        ).replace(REFINED_SLOT, refined_output),
    )


