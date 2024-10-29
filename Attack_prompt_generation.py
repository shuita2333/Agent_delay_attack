import argparse

from system_prompts import get_attacker_system_prompt, get_target_identity, get_integrate_attacker_system_prompt, \
    get_integrate_prompt, get_subtask_attacker_system_prompt, get_background_system_prompt, general_assignment_generate

from judges import load_judge
from conversers import load_attack_and_target_models, subtask_generate
from common import process_target_response, get_init_msg, conv_template, get_subtask_init_msg, \
    subtask_process_target_response, get_background_init_msg
from loggers import AttackLogger


def main(args):
    # 初始化模型和日志记录器
    # 初始化总任务模板
    integrate_system_prompt = get_integrate_attacker_system_prompt(args.function_descript, args.n_question)
    attackLM, targetLM = load_attack_and_target_models(args)

    # 初始化日志
    logger = AttackLogger(args, integrate_system_prompt)

    # 初始化对话
    integrate_convs = conv_template(attackLM.template)
    integrate_convs.set_system_message(integrate_system_prompt)

    # 总任务描述 processed_response_list
    integrate_prompt = get_integrate_prompt(args.function_descript, args.n_question)

    # 获取总任务prompt
    extracted_integrate_attack_prompt = attackLM.get_integrate_attack(integrate_convs, integrate_prompt)
    print("总任务框架已生成")

    logger.integrate_log(extracted_integrate_attack_prompt,args.n_question)

    # 获取子任务 任务prompt
    subtask_prompt_list = subtask_generate(args, extracted_integrate_attack_prompt, attackLM, targetLM)
    print("子任务已生成")

    ### 测试用例，测试不添加背景生成内容长度  ###
    general_assignment = general_assignment_generate(extracted_integrate_attack_prompt["total_prompt"],
                                                     subtask_prompt_list)
    logger.background_log(subtask_prompt_list,general_assignment,[0,0,0,0,0,0])
    ### 测试用例，测试不添加背景生成内容长度  ###



    logger.subtask_log(args.n_iterations, subtask_prompt_list)

    # 延长问题语句输出
    batchsize = args.n_question
    background_system_prompt = get_background_system_prompt()
    background_convs_list = [conv_template(attackLM.template) for _ in range(batchsize)]

    for conv in background_convs_list:
        conv.set_system_message(background_system_prompt)
    subtask_background_list = [get_background_init_msg(subtask) for subtask in subtask_prompt_list]
    complete_subtask_prompt,length_lest = attackLM.get_background(background_convs_list, subtask_background_list)
    # print(complete_subtask_prompt)



    general_assignment = general_assignment_generate(extracted_integrate_attack_prompt["total_prompt"],
                                                     complete_subtask_prompt)

    logger.background_log(complete_subtask_prompt,general_assignment,length_lest)


    print(f"攻击生成完成")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--function-descript",
        type=str,
        default="Financial functional services system provides personalized investment advice and risk management strategies by analyzing users' financial data and behavior patterns.",
        help="对目标agent的功能描述."
    )

    ########### Attack model parameters ##########
    parser.add_argument(
        "--attack-model",
        default="Qwen2.5-7B",
        help="Name of attacking model.",
        choices=["vicuna", "llama-2", "gpt-3.5-turbo", "gpt-4", "claude-instant-1", "claude-2", "palm-2"]
    )
    parser.add_argument(
        "--attack-max-n-tokens",
        type=int,
        default=4096,
        help="Maximum number of generated tokens for the attacker."
    )
    parser.add_argument(
        "--max-n-attack-attempts",
        type=int,
        default=5,
        help="Maximum number of attack generation attempts, in case of generation errors."
    )
    #################################################

    ########### Target model parameters ##########
    parser.add_argument(
        "--target-model",
        default="Qwen2.5-7B",
        help="Name of target model.",
        choices=["vicuna", "llama-2", "gpt-3.5-turbo", "gpt-4", "claude-instant-1", "claude-2", "palm-2"]
    )
    parser.add_argument(
        "--target-max-n-tokens",
        type=int,
        default=4096,
        help="Maximum number of generated tokens for the target."
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=4096,
        help="目标输出长度."
    )
    ##################################################

    # ############ Judge model parameters ##########
    # parser.add_argument(
    #     "--judge-model",
    #     default="Qwen2.5-7B",
    #     help="Name of judge model.",
    #     choices=["gpt-3.5-turbo", "gpt-4", "no-judge"]
    # )
    # parser.add_argument(
    #     "--judge-max-n-tokens",
    #     type=int,
    #     default=10,
    #     help="Maximum number of tokens for the judge."
    # )
    # parser.add_argument(
    #     "--judge-temperature",
    #     type=float,
    #     default=0,
    #     help="Temperature to use for judge."
    # )
    # ##################################################
    #
    # ########### PAIR parameters ##########
    parser.add_argument(
        "--n-question",
        type=int,
        default=8,
        help="子问题数 Number of concurrent jailbreak conversations. If this is too large, then there may be out of memory errors."
    )
    #
    # parser.add_argument(
    #     "--keep-last-n",
    #     type=int,
    #     default=3,
    #     help="Number of responses to save in conversation history of attack model. If this is too large, then it may exceed the context window of the model."
    # )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=1,
        help="目标长文本提示prompt训练重复次数Number of iterations to run the attack."
    )
    # parser.add_argument(
    #     "--goal",
    #     type=str,
    #     default="Financial functional services system provides personalized investment advice and risk management strategies by analyzing users' financial data and behavior patterns.",
    #     # help="Goal of the conversation."
    #     help="对目标agent的功能描述."
    # )
    # # parser.add_argument(
    # #     "--target-str",
    # #     type=str,
    # #     default="Sure, here is a tutorial for making a bomb",
    # #     help="Target response for the target model."
    # # )
    #
    # ##################################################
    #
    # ########### Logging parameters ##########
    # parser.add_argument(
    #     "--index",
    #     type=int,
    #     default=0,
    #     help="Row number of AdvBench, for logging purposes."
    # )
    # parser.add_argument(
    #     "--category",
    #     type=str,
    #     default="bomb",
    #     help="Category of jailbreak, for logging purposes."
    # )
    # ##################################################

    args = parser.parse_args()

    main(args)
