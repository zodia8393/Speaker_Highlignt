from huggingface_hub import login

def login_to_huggingface(token):
    login(token)
    logger.info("Logged into Hugging Face")
