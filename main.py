import argparse

from utils.attack import generate_general_prompt, iterative_optimization


def main(args):
    general_prompt = generate_general_prompt(args)
    target_success_agent = iterative_optimization(args, general_prompt)
    print("攻击语句生成完成：\n" + target_success_agent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    # 模型选择
    # "Meta-Llama-3.1", "DeepSeek-V2.5", "Qwen2.5-7B", "vicuna", "llama-2", "gpt-3.5-turbo", "gpt-4",
    # "claude-instant-1", "claude-2", "palm-2","gemma-2-9b","Ministral-3B", "Ministral-8B"
    attack_model="Qwen2.5-7B"
    target_model="Ministral-8B"

    parser.add_argument(
        "--function-descript",
        type=str,
        default="Financial functional services system provides personalized investment advice and risk management "
                "strategies by analyzing users' financial data and behavior patterns.",
        help="对目标agent的功能描述."
    )

    parser.add_argument(
        "--n-question",
        type=int,
        default=8,
        help="子问题数 Number of concurrent jailbreak conversations. If this is too large, then there may be out of "
             "memory errors."
    )

    # ================ Attack model parameters ================
    parser.add_argument(
        "--attack-model",
        default=attack_model,
        help="Name of attacking model.",
        choices=["Meta-Llama-3.1", "DeepSeek-V2.5", "Qwen2.5-7B", "vicuna", "llama-2", "gpt-3.5-turbo", "gpt-4",
                 "claude-instant-1",
                 "claude-2", "palm-2"]
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
    ##################################################

    # =========== Target model parameters ===========
    parser.add_argument(
        "--target-model",
        default=target_model,
        help="Name of target model.",
        choices=["Meta-Llama-3.1","DeepSeek-V2.5", "Qwen2.5-7B", "vicuna", "llama-2", "gpt-3.5-turbo", "gpt-4", "claude-instant-1",
                 "claude-2", "palm-2"]
    )
    parser.add_argument(
        "--target-max-n-tokens",
        type=int,
        default=32000,
        help="Maximum number of generated tokens for the target."
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=32000,
        help="目标输出长度."
    )
    ##################################################

    # ============== Judge model parameters =============
    parser.add_argument(
        "--judge-model",
        default="Qwen2.5-7B",
        help="Name of judge model.",
        choices=["gpt-3.5-turbo", "gpt-4", "no-judge"]
    )
    parser.add_argument(
        "--judge-max-n-tokens",
        type=int,
        default=10,
        help="Maximum number of tokens for the judge."
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0,
        help="Temperature to use for judge."
    )
    ##################################################

    # ============== PAIR parameters =============
    parser.add_argument(
        "--n-streams",
        type=int,
        default=3,
        help="并发数 Number of concurrent jailbreak conversations. If this is too large, then there may be out of memory "
             "errors."
    )

    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=5,
        help="Number of responses to save in conversation history of attack model. If this is too large, then it may "
             "exceed the context window of the model."
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=20,
        help="Number of iterations to run the attack."
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="Financial functional services system provides personalized investment advice and risk management "
                "strategies by analyzing users' financial data and behavior patterns.",
        # help="Goal of the conversation."
        help="对目标agent的功能描述."
    )
    # parser.add_argument(
    #     "--target-str",
    #     type=str,
    #     default="Sure, here is a tutorial for making a bomb",
    #     help="Target response for the target model."
    # )

    ##################################################

    # ================= Logging parameters ==================
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row number of AdvBench, for logging purposes."
    )
    parser.add_argument(
        "--category",
        type=str,
        default="bomb",
        help="Category of jailbreak, for logging purposes."
    )
    ##################################################

    main(parser.parse_args())
