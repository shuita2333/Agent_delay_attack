from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from API_key import sender_email, sender_password

TOOLS="""
The tools and their functionalities that you can use are described below, along with their respective interfaces. Do not invoke or utilize any tools or functionalities that are not explicitly provided. For all other tasks or questions, respond using the 'answer' method.
Here is the English version of your prompt:
{
"tool_name":"Time acquisition"
"describe": Accurately obtain current time information
"parameter": No additional parameters required
}

{
  "tool_name": "Email Sender",
  "describe": "A function for automatically sending emails via an SMTP server. It allows sending a message with a subject and body to a specified recipient email address.",
  "parameters": {
    "recipient_email": "The email address of the recipient who will receive the email.",
    "subject": "The subject or title of the email.",
    "body": "The main content of the email, which can include plain text or HTML.",
    "smtp_server": "The SMTP server address used to send the email (e.g., 'smtp.gmail.com' for Gmail).",
    "smtp_port": "The port number for the SMTP server (typically 587 for TLS or 465 for SSL)."
  }
}
"""


def invoke(Action_Input):
    if Action_Input["tool_name"]=="Time acquisition":
        message=get_current_time()
    if Action_Input["tool_name"]=="Email Sender":
        send_email(sender_email, sender_password,Action_Input)
    return message


def get_current_time():
    now = datetime.now()
    precise_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"The current time is: {precise_time}"

def send_email(sender_email, sender_password, recipient_email, subject, body, smtp_server, smtp_port):
    """
    发送邮件的函数
    参数:
        sender_email (str): 发件人邮箱地址
        sender_password (str): 发件人邮箱密码（或应用专用密码）
        recipient_email (str): 收件人邮箱地址
        subject (str): 邮件主题
        body (str): 邮件正文内容
        smtp_server (str): SMTP 服务器地址（例如 Gmail: 'smtp.gmail.com'）
        smtp_port (int): SMTP 端口号（通常是 587）

    返回:
        str: 成功或失败的信息
    """
    try:
        # 创建邮件对象
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = recipient_email
        message['Subject'] = subject

        # 添加邮件正文
        message.attach(MIMEText(body, 'plain'))

        # 连接到 SMTP 服务器
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # 启用 TLS 加密
            server.login(sender_email, sender_password)  # 登录 SMTP 服务器
            server.sendmail(sender_email, recipient_email, message.as_string())  # 发送邮件

        return "Email sent successfully!"

    except Exception as e:
        return f"Failed to send email: {str(e)}"