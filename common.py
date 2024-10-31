from fastchat.model import get_conversation_template

def conv_template(template_name):
    '''
    获取对应的对话模板
    :param template_name:
    :return:
    '''
    template = get_conversation_template(template_name)
    if template.name == 'llama-2':
        template.sep2 = template.sep2.strip()
    return template
