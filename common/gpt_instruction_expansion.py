import numpy as np

try:
    from PIL import Image
except Exception as e:
    print(f"Error: {e}")
import base64
import io
import os

# Template for prompt, use get_prompt_head(n) to get the prompt with n new instructions
PROMPT_HEAD_TEMPLATE = """
You are a human manipulation task instruction re-writer. Given \\
1. an original task instruction \\
2. three image from human execution video of this task \\
your goal is to re-write {n} new instructions which: \\
1. the task should be exactly the same with the original instruction. \\
2. you may make more details for some instructions with images hint, e.g. which color/material/type of objects, under which backgrounds, detailed actions, use which hand, etc. \\
3. keep the diversity of your re-written instructions, e.g. some is short and concise (even only serveral words), some is long and detailed. \\
4. no words like 'careful', 'seriesly' used in the instructions. \\
your output format should be like this: \\
[1] new_instruction_1. \\
[2] new_instruction_2. \\
[3] new_instruction_3. \\
... \\
[{n_minus_1}] new_instruction_{n_minus_1}. \\
[{n}] new_instruction_{n}. \\
Please follow the format strictly, do not add any other text and content. Make sure the task is the same for new and original instruction, and follows the content of images strictly. \\
Input images:
"""

def get_prompt_head(n_new_instructions: int) -> str:
    """Get the prompt head with the specified number of new instructions."""
    return PROMPT_HEAD_TEMPLATE.format(
        n=n_new_instructions,
        n_minus_1=n_new_instructions - 1
    )



def img_to_base64(img_array):
    """Convert numpy image array to base64 string"""
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_response(client, prompt, model="gpt-4o"):
    n_trial = 3
    response = None
    for _ in range(n_trial):
        try:
            response = client.chat.completions.create(
                messages=prompt,
                model=model
            )
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            continue
    if response is None:
        return None
    return response.choices[0].message.content


def get_openai_instruction_expansion_episodes(episode,
                         image_key,
                         n_image,
                         instruction_key,
                         client,
                         embodiment,
                         n_new_instructions=10,
                         fill_instruction_length=100):

    images = episode[image_key]
    n_frames = len(images)
    # image_idx_list = np.random.choice(n_frames, n_image, replace=False)
    image_idx_list = np.linspace(0, n_frames - 1, n_image, dtype=int)
    images = [images[i] for i in image_idx_list]
    instruction_old = episode[instruction_key][0]

    # Prepare the prompt for the AI model
    prompt_head = get_prompt_head(n_new_instructions)
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": prompt_head.replace("human", embodiment)
                }
            ]
        },
        {
            "role": "user",
            "content": [
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_to_base64(image)}",
                        }
                    }
                    for image in images
                ],
                {
                    "type": "text",
                    "text": "Original instruction: " + instruction_old + ". Output:"
                }
            ]
        }
    ]

    # Get the response from the AI model
    response = get_response(client, chat_prompt, model="gpt-5.4")
    if response is None:
        print("Failed to get response.")
        instructions = [instruction_old.ljust(fill_instruction_length)]
    else:
        # get the instructions list from the response
        instructions = response.split("\n")
        instructions = [inst.strip() for inst in instructions if inst.strip()]
        instructions = [inst.split("] ")[-1].strip() for inst in instructions]
        # fill each instruction to length of 200
        instructions = [inst.ljust(fill_instruction_length) for inst in instructions]
    instruction_old = instruction_old.ljust(fill_instruction_length)

    instruction_list = []
    for _ in range(n_frames):
        random_idx = np.random.randint(5)
        if random_idx == 0:
            instruction_list.append(instruction_old)
        else:
            instruction_new = instructions[np.random.randint(len(instructions))]
            instruction_list.append(instruction_new)

    episode[instruction_key] = np.array(instruction_list)
    return episode, instructions


def get_openai_client(api_version="2024-12-01-preview"):
    azure_endpoint = os.environ.get("AZURE_ENDPOINT")
    azure_api_key = os.environ.get("AZURE_API_KEY")
    api_version = os.environ.get("AZURE_API_VERSION", api_version)

    if not azure_endpoint or not azure_api_key:
        raise ValueError(
            "Azure OpenAI credentials are not configured. "
            "Please set AZURE_ENDPOINT and AZURE_API_KEY."
        )

    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=api_version,
    )


def get_qwen_client():
    """Return a Qwen-compatible OpenAI client using DashScope.

    The client connects to DashScope's OpenAI-compatible endpoint and
    uses the DASHSCOPE_API_KEY environment variable for authentication.
    """
    from openai import OpenAI  # local import to avoid changing existing imports

    # api_key = os.getenv("DASHSCOPE_API_KEY")
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable is not set.")

    return OpenAI(
        api_key=api_key,
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )


def get_instruction_expansion_episodes_qwen(
    episode,
    image_key,
    n_image,
    instruction_key,
    client,
    embodiment,
    n_new_instructions: int = 10,
    fill_instruction_length: int = 100,
    # model: str = "qwen-plus",
    model: str = "doubao-seed-1-6-vision-250815",
):
    """Qwen-based variant of get_instruction_expansion_episodes.

    The behavior mirrors get_instruction_expansion_episodes but sends the
    request to a Qwen model (default: "qwen-plus") via an OpenAI-compatible
    client, typically created by get_qwen_client().
    """

    images = episode[image_key]
    n_frames = len(images)
    image_idx_list = np.linspace(0, n_frames - 1, n_image, dtype=int)
    images_sel = [images[i] for i in image_idx_list]
    instruction_old = episode[instruction_key][0]

    # Prepare the prompt for the AI model
    prompt_head = get_prompt_head(n_new_instructions)
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": prompt_head.replace("human", embodiment),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_to_base64(image)}",
                        },
                    }
                    for image in images_sel
                ],
                {
                    "type": "text",
                    "text": "Original instruction: " + instruction_old + ". Output:",
                },
            ],
        },
    ]

    response = get_response(client, chat_prompt, model=model)
    if response is None:
        print("Failed to get response from Qwen.")
        instructions = [instruction_old.ljust(fill_instruction_length)]
    else:
        lines = response.split("\n")
        lines = [ln.strip() for ln in lines if ln.strip()]
        instructions = [ln.split("] ", 1)[-1].strip() for ln in lines]
        instructions = [inst.ljust(fill_instruction_length) for inst in instructions]

    instruction_old_padded = instruction_old.ljust(fill_instruction_length)

    instruction_list = []
    for _ in range(n_frames):
        random_idx = np.random.randint(5)
        if random_idx == 0 or not instructions:
            instruction_list.append(instruction_old_padded)
        else:
            instruction_new = instructions[np.random.randint(len(instructions))]
            instruction_list.append(instruction_new)

    episode[instruction_key] = np.array(instruction_list)
    return episode, instructions
