import torch

from PIL import Image

def delete_char(sentence):
    first_comma_index = sentence.find(',')
    last_comma_index = sentence.rfind(',')

    if first_comma_index != -1 and last_comma_index != -1:
        prompt = sentence[first_comma_index + 2: last_comma_index]

        return prompt
    else:
        raise Exception("Wrong Data Format!")


def load_prompt(path, prompt_version):
    if prompt_version == 'pick':
        prompts = []
        seeds = []
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.readlines()

            for row in contents:
                prompt = delete_char(row)
                seed = row.split(' ')[-1]

                prompts.append(prompt)
                seeds.append(seed)

        return prompts, seeds
    

def load_pick_prompt(path):
    data_dict = {}

    with open(path, 'r', encoding='utf-8') as file:
        content = file.readlines()

        for row in content:
            row = eval(row)
            data_dict[row['index']] = row['caption']
        
    return data_dict


def load_pick_discard_prompt(path):
    bad_indexes = []

    data_dict = {}

    count = 0

    with open(path, 'r', encoding='utf-8') as file:
        content = file.readlines()

        for row in content:
            row = eval(row)

            # if all(col[0] < col[1] for col in zip(*[row['original_score_list'], row['optimized_score_list']])):
            #     data_dict[row['index']] = row['caption']
            if row['original_score_list'][0] < row['optimized_score_list'][0]:
                data_dict[row['index']] = row['caption']

            else:
                bad_indexes.append(row['index'])
                count += 1
        
    print(f"bad sample rate {round(count * 100 / len(content), 2)}%")

    return data_dict, bad_indexes



def infer_example(images, prompt, condition, clip_model, clip_processor, tokenizer, device):
    def _process_image(image):
        if isinstance(image, dict):
            image = image["bytes"]
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        if isinstance(image, str):
            image = Image.open( image )
        image = image.convert("RGB")
        pixel_values = clip_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values
    
    def _tokenize(caption):
        input_ids = tokenizer(
            caption,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids
    
    image_inputs = torch.concatenate([_process_image(images[0]).to(device), _process_image(images[1]).to(device)])
    text_inputs = _tokenize(prompt).to(device)
    condition_inputs = _tokenize(condition).to(device)

    with torch.no_grad():
        text_features, image_0_features, image_1_features = clip_model(text_inputs, image_inputs, condition_inputs)
        image_0_features = image_0_features / image_0_features.norm(dim=-1, keepdim=True)
        image_1_features = image_1_features / image_1_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_0_scores = clip_model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_0_features))
        image_1_scores = clip_model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_1_features))
        scores = torch.stack([image_0_scores, image_1_scores], dim=-1)
        probs = torch.softmax(scores, dim=-1)[0]

    return probs.cpu().tolist()
