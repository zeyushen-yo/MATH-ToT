import json
import os
import csv
import pandas as pd
import openai
client = openai.OpenAI()
from tqdm import tqdm
from tot.aggregated_skills import aggregated_skills
from tot.tasks.base import DATA_PATH

topics = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

aggregated_skills = aggregated_skills

with open("skill_examples/math_train_with_skill.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['problem', 'solution', 'skill'])
    for topic in topics:
        path = DATA_PATH + "/MATH/train/" + topic + "/"
        for filename in tqdm(os.listdir(path)):
            full_path = path + filename
            with open(full_path, "r") as f:
                data = json.load(f)

                problem = data["problem"]
                solution = data["solution"]
                prompt = f"Here is a list of skills:\n {aggregated_skills} \n for solving mathematical problems."
                prompt += f"Consider this problem from {topic}. Label this problem with a mathematical skill from this list. You must name the skill in exactly the same way as it appears in the list.\n"
                prompt += "problem: " + problem + "\n\n Your answer should be as follows: \n <name of the skill> \n"
                
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": prompt},
                    ]
                )
            
                content = completion.choices[0].message.content
                content = content.split("\n")
                skill = content[0].strip()
                skill = skill.replace(",", " | ")
                print(skill)

                writer.writerow([problem, solution, skill])
                f.flush()

skills_to_example = {}
df = pd.read_csv("math_train_with_skill.csv")

for i in range(len(df)):
    if ":" in df.loc[i, "skill"]:
        skill = df.loc[i, "skill"].split(":")[1].strip()
    else:
        skill = df.loc[i, "skill"].strip()
    if skill not in skills_to_example:
        skills_to_example[skill] = []
    skills_to_example[skill].append([df.loc[i, "problem"], df.loc[i, "solution"]])

for s in skills_to_example:
    in_list = False
    for agg_s in aggregated_skills:
        if s in aggregated_skills[agg_s]:
            in_list = True
    if in_list:
        with open(f"skill_examples/math_train_{s}_with_examples.jsonl", "w", encoding="utf-8") as file:
            for i in range(len(skills_to_example[s])):
                problem = skills_to_example[s][i][0]
                solution = skills_to_example[s][i][1]
                data = {"problem": problem, "solution": solution}
                file.write(json.dumps(data, ensure_ascii=False) + "\n")