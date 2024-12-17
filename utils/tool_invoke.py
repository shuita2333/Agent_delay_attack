from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from API_key import sender_email, sender_password

TOOLS = """
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
    message = ""
    if Action_Input["tool_name"] == "Time acquisition":
        message = get_current_time()
    if Action_Input["tool_name"] == "Email Sender":
        message = send_email(sender_email, sender_password, Action_Input)
    return message


def get_current_time():
    now = datetime.now()
    precise_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"The current time is: {precise_time}"


def send_email(sender_email, sender_password, recipient_email, subject, body, smtp_server, smtp_port):
    """
    Function for sending emails
    Parameters:
        sender_email (str): sender's email address
        sender_password (str): sender's email password (or application-specific password)
        recipient_email (str): recipient's email address
        subject (str): email subject
        body (str): email body
        smtp_server (str): SMTP server address (e.g. Gmail: 'smtp.gmail.com')
        smtp_port (int): SMTP port number (usually 587)

    Returns:
        str: success or failure information
    """
    try:
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = recipient_email
        message['Subject'] = subject

        message.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())

        return "Email sent successfully!"

    except Exception as e:
        return f"Failed to send email: {str(e)}"
