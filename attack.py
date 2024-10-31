from agents.AgentFactory import AgentFactory, load_optimize_agents
from messages import get_general_message
from loggers import AttackLogger

logger = AttackLogger()


def generate_general_prompt(args):
    integrate_agent = AgentFactory.get_factory('IntegrateAgent', args)
    integrate_agent_init_message = integrate_agent.get_init_msg()
    integrate_agent_conv_list = integrate_agent.get_conv_list(1)
    integrate_agent_processed_response_list = [integrate_agent_init_message for _ in range(1)]
    # 得到integrate_agent的结果
    extracted_integrate_agent_list = integrate_agent.get_response(integrate_agent_conv_list,
                                                                  integrate_agent_processed_response_list)
    print("总任务框架已生成")
    # 分离Json
    integrate_agent_total_prompt = extracted_integrate_agent_list[0]["total_prompt"]
    integrate_agent_subtask_question = extracted_integrate_agent_list[0]["subtask_question"]

    # 生成子问题
    subtask_prompt_list = integrate_agent.get_sub_problems(integrate_agent_total_prompt,
                                                           integrate_agent_subtask_question)
    print("子任务已生成")
    # 得到子问题的回复
    subtask_answer_list = integrate_agent.get_sub_answers(subtask_prompt_list)
    print("子回答已生成")
    general_prompt = get_general_message(integrate_agent_total_prompt,
                                         subtask_answer_list)
    logger.log(Description="sub task finished",
               iteration='1',
               integerateAgent_total_prompt=integrate_agent_total_prompt,
               integerateAgent_subtask_question=integrate_agent_subtask_question,
               subtask_prompt_list=subtask_prompt_list,
               subtask_answer_list=subtask_answer_list,
               general_prompt=general_prompt
               )
    return general_prompt


def iterative_optimization(args, general_prompt):
    # target_lm = load_target_model(args)
    target_agent, method_agent, judge_agent = load_optimize_agents(args)

    batch_size = args.n_streams

    # 不同Agent的对话模板
    method_agent_conv_list = method_agent.get_conv_list(batch_size)
    judge_agent_conv_list = judge_agent.get_conv_list(batch_size)
    target_agent_conv_list = target_agent.get_conv_list(batch_size)

    # methodAgent和integrateAgent的Prompt
    method_agent_processed_response_list = [method_agent.get_init_message(general_prompt) for _ in range(batch_size)]

    method_agent_suggestion_list = []
    method_agent_pre_prompt = []
    method_agent_post_prompt = []
    # 开始对话
    for iteration in range(1, args.n_iterations + 1):
        print(f"""\n{'=' * 36}\nIteration: {iteration}\n{'=' * 36}\n""")
        if iteration > 1:
            # 如果不是第一次输出，就采用process_suggestion为Agent提供建议
            method_agent_processed_response_list = [
                method_agent.process_suggestion(Prepare_prompt, general_prompt, Post_prompt, suggestion) for
                Prepare_prompt, Post_prompt, suggestion in
                zip(method_agent_pre_prompt, method_agent_post_prompt, method_agent_suggestion_list)]

        # 获得改进后的策略和内容
        extracted_method_agent_list = method_agent.get_response(method_agent_conv_list,
                                                                method_agent_processed_response_list)

        print("Finished getting agent prompts.")

        # 提取 methodAgent的改进prompt
        method_agent_pre_prompt = [attack["Prepare_prompt"] for attack in extracted_method_agent_list]
        method_agent_post_prompt = [attack["Post_prompt"] for attack in extracted_method_agent_list]

        # 得到综合后的结果
        review_agent_synthesize_list = [Prepare_prompt + general_prompt + Post_prompt for Prepare_prompt, Post_prompt in
                                        zip(method_agent_pre_prompt, method_agent_post_prompt)]
        print("得到了综合后的结果")

        # 获得目标响应
        target_information_list = target_agent.get_response(target_agent_conv_list,
                                                            review_agent_synthesize_list)

        target_response_list = [target_information_list[i]['content_str'] for i in range(batch_size)]
        target_response_length = [target_information_list[i]['content_length'] for i in range(batch_size)]

        print("得到了target agent的输出")

        # 根据已有的信息，生成judgeAgent的prompt
        judged_content = [judge_agent.judge_content(method_agent_pre_prompt[i],
                                                    general_prompt,
                                                    method_agent_post_prompt[i],
                                                    target_response_list[i]) for i in range(batch_size)]

        # 得到judgeAgent给出的建议
        extracted_judge_agent_list = judge_agent.get_response(judge_agent_conv_list, judged_content)
        judge_agent_evaluate = [attack["evaluate"] for attack in extracted_judge_agent_list]

        print("得到judgeAgent给出的建议")

        method_agent_suggestion_list = judge_agent_evaluate

        logger.log(iteration=iteration,
                   method_agent_pre_prompt=method_agent_pre_prompt,
                   method_agent_post_prompt=method_agent_post_prompt,
                   reviewAgent_synthesize_list=review_agent_synthesize_list,
                   target_response_list=target_response_list,
                   target_response_length=target_response_length,
                   judgeAgent_evaluate=judge_agent_evaluate)

        # 早停准则
        if any([length >= args.target_length * 0.9 for length in target_response_length]):
            print("Found a jailbreak. Exiting.")
            break
