import os, sys, json
import prompts_dir as prompts_dir


import requests
import json
import time

def test_inference(prompt_now):
    url = "http://localhost:8001/process"
    test_messages =  [
    # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "system", "content": "You are a helpful chatbot!!"},
    # {"role": "system", "content": "Be a good chatbot. Answer everything but in an Obamaish tone.!"},
    #########################
    # {"role": "user", "content": "Who are you?"},
    {"role": "user", "content": prompt_now},
    # {"role": "user", "content": "What is the capital of France?"},
    # {"role": "user", "content": "Who is Thanos?"},
    ]
    # print("test_messages: ", test_messages)
    payload = {
        "prompt": None,
        "message_list": test_messages,
        # "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        # "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "num_samples": 1,  # Required field
        "max_new_tokens": 2000,
        "top_k": 50,
        "temperature": 0.7,
        "seed": 42,
        "obj_creation_time": time.time(),  # Required field
        "prompt_num_tokens": None
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))
        print("Output is: ", response.json()['model_output'])
        # return response.json()['model_output']
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error details: {response.text}")
        print(f"Status code: {response.status_code}")
        return None

def get_llm_response(query):
    return test_inference(query)
    return ""
angle_vals = [0, 15, 30, 40, 45, 50, 55, 60, 75, 90]
intensity_vals = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 5.0]
save_dir = "saved_gens"
for angle in angle_vals:
    for intensity in intensity_vals:
        prompt_now = prompts_dir.polar_gen_prompt
        save_path = os.path.join(save_dir, f"prompt_{angle}_{intensity}.txt")
        if os.path.exists(save_path):
            continue
        prompt_now = f"""
{prompt_now}

# Details of the review to generate:
Theta: {angle}
Radius: {intensity}
"""
        try:
            llm_ans = get_llm_response(prompt_now)
            if llm_ans is None:
                print("Error in response")
            # extract review from between <review> tags
            start_tag = "<review>"
            end_tag = "</review>"
            assert(start_tag in llm_ans['model_output'])
            assert(end_tag in llm_ans['model_output'])
            # start_idx = llm_ans.find(start_tag)
            # end_idx = llm_ans.find(end_tag)
            # review = llm_ans[start_idx + len(start_tag):end_idx]
            # with open(save_path, "w") as f:
            #     write_ans = f"{review}\n###########\n#######\n{llm_ans}"
            #     f.write(write_ans)
            with open(save_path, "w") as f:
                json.dump(llm_ans, f, indent=2)
                print(llm_ans['model_output'])
            print(f"Saved review to {save_path}")
        except:
            print("Error in generating review")
            continue
    # break
        
        
