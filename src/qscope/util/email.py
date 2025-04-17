import yagmail  # install yagmail[all]

# TODO try yag.send(receiver_emails, subject, contents, headers={"Reply-To":f"{reply_email}"})
# https://github.com/kootenpv/yagmail/issues/33


def send_mail(
    sender: str, receiver: str, body: str, subject: str = "QScope notification"
):
    """You probably want to run a test mail first, and add the username and password to keyring.
    To set up an account with gmail:
    (1) Set up a new google account
    (2) Add 2-step verification
    (3) Create an app password (https://support.google.com/accounts/answer/185833)
    (4) Run this function as a test, and add the username and password to keyring.
    Simples!
    """
    yag = yagmail.SMTP(sender)
    yag.send(
        to=receiver,
        subject=subject,
        contents=body,
    )


def send_mail_attachments(
    sender: str,
    receiver: str,
    body: str,
    attachments: list[str],
    subject="QScope notification",
):
    """See docs for `send_mail`. Attachments is a list of paths to files to attach."""
    yag = yagmail.SMTP(sender)
    yag.send(
        to=receiver,
        subject=subject,
        contents=body,
        attachments=attachments,
    )
