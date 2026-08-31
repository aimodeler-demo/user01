# ======================================
# Generate Dataset
#
# Instructions:
# This code will generate a dataset of
# 50 question-and-answer pairs from
# user-selected categories that can be
# used to fine-tune an LLM model with
# the OpenAI fine-tuning platform.
# There are two category lists included
# containing Science and Arts categories
# but these lists can be easily changed.
# The dataset is in JSONL format and is
# validated before the job is accepted.
# ======================================

from openai import OpenAI
import json

client = OpenAI()

CATEGORIES = [
    "SCIENCE",
    "MATHEMATICS",
    "ENGINEERING",
    "TECHNOLOGY",
    "MEDICINE"
]

# CATEGORIES = [
#     "ART HISTORY",
#     "ENGLISH LITERATURE",
#     "WESTERN PHILOSOPHY",
#     "CLASSICAL MUSIC",
#     "WORLD HISTORY"
# ]

dataset = []

for category in CATEGORIES:

    prompt = f"""
Generate EXACTLY 10 unique educational question–answer pairs
for the category: {category}.

Rules:
- Each question must be clearly different
- No paraphrases of the same question
- Each answer must be a clear explanation,
  of at least 2–4 sentences long
- Answers must be factual and suitable for
  undergraduate education

The dataset should be returned as a valid JSON object:

Return ONLY valid JSON in this exact format:
[
  {{
    "messages": [
      {{
        "role": "user",
        "content": "Question text here"
      }},
      {{
        "role": "assistant",
        "content": "Answer text here"
      }}
    ]
  }}
]
"""

    print(f"\nGenerating dataset for {category}...", flush=True)
    
    response = client.responses.create(
        model="gpt-4.1",
        temperature=0,
        input=[{"role": "user", "content": prompt}]
    )

    items = json.loads(response.output_text)

    if len(items) != 10:
        raise ValueError(
            f"Expected 10 examples for {category}, got {len(items)}"
        )

    print(f"Completed {category} category with {len(items)} examples.", flush=True)
    
    dataset.extend(items)

if len(dataset) != 50:
    raise ValueError(
        f"Expected 50 examples, got {len(dataset)}"
    )

with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

print(f"\nDataset created with {len(dataset)} examples.\n")

print("First record from new dataset:\n")

with open("dataset.jsonl", "r", encoding="utf-8") as f:
    print(f.readline())


# UPLOAD THE TRAINING FILE

from openai import OpenAI

client = OpenAI()

response = client.files.create(
  file=open("dataset.jsonl", "rb"),
  purpose="fine-tune"
)

print(f"File uploaded successfully. ID: {response.id}")


MODEL = "gpt-4.1-2025-04-14"

job = client.fine_tuning.jobs.create(
  training_file=response.id,

  model=MODEL,
  hyperparameters={
    "n_epochs": 3
  }
)

job_id = job.id

print(f"Fine-tuning job started. Job ID: {job_id}")


# DISPLAY PROGRESS OF RECENT FINE-TUNING JOBS

events = client.fine_tuning.jobs.list_events(
    fine_tuning_job_id=job.id,
    limit=10
)

for event in events:
    print(f"[{event.created_at}] {event.message}")


# CHECK THE STATUS OF FINE-TUNING JOB

import time

print(f"Checking status for job: {job_id}...")

while True:

    job = client.fine_tuning.jobs.retrieve(job_id)

    if job.status == "succeeded":

        model_name = job.fine_tuned_model

        print(f"\nFine-tuning succeeded.")
        print(f"Fine-tuned model: {model_name}")
        break

    elif job.status == "failed":

        print(f"\nFine-tuning failed: {job.error.message}")
        raise RuntimeError("Fine-tuning job failed.")

    else:

        print(f"Status: {job.status}... waiting 60s")
        time.sleep(60)



# CANCEL ANY ACTIVE FINE-TUNING JOBS

jobs = client.fine_tuning.jobs.list(limit=10)

active_jobs = [
    job for job in jobs.data
    if job.status in ["validating_files", "queued", "running"]
]

if active_jobs:

    print("Active fine-tuning jobs:")

    for job in active_jobs:
        print(f"- {job.id} (Status: {job.status})")



# CANCEL ANY ACTIVE FINE-TUNING JOBS

jobs = client.fine_tuning.jobs.list(limit=10)

active_jobs = [
    job for job in jobs.data
    if job.status in ["validating_files", "queued", "running"]
]

if active_jobs:

    print("Active fine-tuning jobs:")

    for job in active_jobs:
        print(f"- {job.id} (Status: {job.status})")

# Run code above for job status only 
# or whole section to cancel jobs

    choice = input("\nCancel all active jobs? (y/n): ").strip().lower()

    if choice == "y":

        for job in active_jobs:
            client.fine_tuning.jobs.cancel(job.id)
            print(f"Cancelled job: {job.id}")

        print("Fine-tuning capacity available for new job.")

    else:
        print("No jobs cancelled.")

else:
    print("No active fine-tuning jobs.")

user = "user01"

import os
import shutil
import time
from git import Repo

file_name = "AI_Modeler_LLM_Demo_01.py"
repo_dir = f"../{user}/repos"
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError("GITHUB_TOKEN environment variable is not set.")

url = f"https://x-access-token:{token}@github.com/aimodeler-demo/{user}.git"

if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)

repo = Repo.clone_from(url, repo_dir)

with open(file_name, "a") as f:
    f.write(f"\n# Main Sync: {time.ctime()}")

shutil.copy2(file_name, os.path.join(repo_dir, file_name))

repo.git.add(all=True)
commit_message = f"Auto-update main: {time.ctime()}"
repo.index.commit(commit_message)

print("Attempting to push to main branch.")
origin = repo.remote(name='origin')

try:
    origin.push(refspec='HEAD:main', force=True)
    print("Main branch updated successfully.")
except Exception as e:
    print(f"Push to main branch failed: {e}")


user = "user01"

# ======================================
# Restore Repository
#
# Instructions:
# This will sync the contents of the
# local repo with those of the remote
# GitHub repo and is extremely useful
# when there a large number of files.
# ======================================

import os
from git import Repo

repo_dir = f"../{user}/repos"
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError("GITHUB_TOKEN environment variable is not set.")

url = f"https://x-access-token:{token}@github.com/aimodeler-demo/{user}.git"

if not os.path.exists(repo_dir):
    repo = Repo.clone_from(url, repo_dir)
else:
    repo = Repo(repo_dir)

origin = repo.remote(name="origin")
origin.fetch()
repo.git.reset("--hard", "origin/main")  
repo.git.clean("-fd")

print(f"Local repo at {repo_dir} has been reset to origin/main")


user = "user01"

import os
import shutil
import time
from git import Repo

file_name = "AI_Modeler_LLM_Demo_01.py"
repo_dir = f"../{user}/repos"
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError("GITHUB_TOKEN environment variable is not set.")

url = f"https://x-access-token:{token}@github.com/aimodeler-demo/{user}.git"

if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)

repo = Repo.clone_from(url, repo_dir)

with open(file_name, "a") as f:
    f.write(f"\n# Main Sync: {time.ctime()}")

shutil.copy2(file_name, os.path.join(repo_dir, file_name))

repo.git.add(all=True)
commit_message = f"Auto-update main: {time.ctime()}"
repo.index.commit(commit_message)

print("Attempting to push to main branch.")
origin = repo.remote(name='origin')

try:
    origin.push(refspec='HEAD:main', force=True)
    print("Main branch updated successfully.")
except Exception as e:
    print(f"Push to main branch failed: {e}")


import os
token = os.getenv('GITHUB_TOKEN')
print(token)

user = "user01"

# ======================================
# Commit History
#
# Instructions:
# This short code snippet displays the
# commit history of a repository file.
# ======================================

import datetime
from git import Repo

file_name = "AI_Modeler_LLM_Demo_01.py"
repo_dir = f"../{user}/repos"

repo = Repo(repo_dir)
commits = repo.iter_commits(paths=file_name)

print(f"History for {file_name}:")
for c in commits:
    date = datetime.datetime.fromtimestamp(c.committed_date).strftime('%Y-%m-%d %H:%M')
    print(f"{date} | {c.hexsha[:7]} | {c.message.strip()}")


user = "user01"

import os
import shutil
import time
from git import Repo

file_name = "AI_Modeler_LLM_Demo_01.py"
repo_dir = f"../{user}/repos"
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError("GITHUB_TOKEN environment variable is not set.")

url = f"https://x-access-token:{token}@github.com/aimodeler-demo/{user}.git"

if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)

repo = Repo.clone_from(url, repo_dir)

with open(file_name, "a") as f:
    f.write(f"\n# Main Sync: {time.ctime()}")

shutil.copy2(file_name, os.path.join(repo_dir, file_name))

repo.git.add(all=True)
commit_message = f"Auto-update main: {time.ctime()}"
repo.index.commit(commit_message)

print("Attempting to push to main branch.")
origin = repo.remote(name='origin')

try:
    origin.push(refspec='HEAD:main', force=True)
    print("Main branch updated successfully.")
except Exception as e:
    print(f"Push to main branch failed: {e}")


user = "user01"

# ======================================
# Restore Repository
#
# Instructions:
# This will sync the contents of the
# local repo with those of the remote
# GitHub repo and is extremely useful
# when there a large number of files.
# ======================================

import os
from git import Repo

repo_dir = f"../{user}/repos"
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError("GITHUB_TOKEN environment variable is not set.")

url = f"https://x-access-token:{token}@github.com/aimodeler-demo/{user}.git"

if not os.path.exists(repo_dir):
    repo = Repo.clone_from(url, repo_dir)
else:
    repo = Repo(repo_dir)

origin = repo.remote(name="origin")
origin.fetch()
repo.git.reset("--hard", "origin/main")  
repo.git.clean("-fd")

print(f"Local repo at {repo_dir} has been reset to origin/main")


user = "user01"

import os
import shutil
import time
from git import Repo

file_name = "AI_Modeler_LLM_Demo_01.py"
repo_dir = f"../{user}/repos"
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError("GITHUB_TOKEN environment variable is not set.")

url = f"https://x-access-token:{token}@github.com/aimodeler-demo/{user}.git"

if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)

repo = Repo.clone_from(url, repo_dir)

with open(file_name, "a") as f:
    f.write(f"\n# Main Sync: {time.ctime()}")

shutil.copy2(file_name, os.path.join(repo_dir, file_name))

repo.git.add(all=True)
commit_message = f"Auto-update main: {time.ctime()}"
repo.index.commit(commit_message)

print("Attempting to push to main branch.")
origin = repo.remote(name='origin')

try:
    origin.push(refspec='HEAD:main', force=True)
    print("Main branch updated successfully.")
except Exception as e:
    print(f"Push to main branch failed: {e}")


user = "user01"

# ======================================
# Restore Repository
#
# Instructions:
# This will sync the contents of the
# local repo with those of the remote
# GitHub repo and is extremely useful
# when there a large number of files.
# ======================================

import os
from git import Repo

repo_dir = f"../{user}/repos"
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError("GITHUB_TOKEN environment variable is not set.")

url = f"https://x-access-token:{token}@github.com/aimodeler-demo/{user}.git"

if not os.path.exists(repo_dir):
    repo = Repo.clone_from(url, repo_dir)
else:
    repo = Repo(repo_dir)

origin = repo.remote(name="origin")
origin.fetch()
repo.git.reset("--hard", "origin/main")  
repo.git.clean("-fd")

print(f"Local repo at {repo_dir} has been reset to origin/main")


user = "user01"

import os
import shutil
import time
from git import Repo

file_name = "AI_Modeler_LLM_Demo_02.py"
repo_dir = f"../{user}/repos"
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError("GITHUB_TOKEN environment variable is not set.")

url = f"https://x-access-token:{token}@github.com/aimodeler-demo/{user}.git"

if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)

repo = Repo.clone_from(url, repo_dir)

with open(file_name, "a") as f:
    f.write(f"\n# Main Sync: {time.ctime()}")

shutil.copy2(file_name, os.path.join(repo_dir, file_name))

repo.git.add(all=True)
commit_message = f"Auto-update main: {time.ctime()}"
repo.index.commit(commit_message)

print("Attempting to push to main branch.")
origin = repo.remote(name='origin')

try:
    origin.push(refspec='HEAD:main', force=True)
    print("Main branch updated successfully.")
except Exception as e:
    print(f"Push to main branch failed: {e}")


user = "user01"

import os
import shutil
import time
from git import Repo

file_name = "AI_Modeler_LLM_Demo_01.py"
repo_dir = f"../{user}/repos"
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError("GITHUB_TOKEN environment variable is not set.")

url = f"https://x-access-token:{token}@github.com/aimodeler-demo/{user}.git"

if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)

repo = Repo.clone_from(url, repo_dir)

with open(file_name, "a") as f:
    f.write(f"\n# Main Sync: {time.ctime()}")

shutil.copy2(file_name, os.path.join(repo_dir, file_name))

repo.git.add(all=True)
commit_message = f"Auto-update main: {time.ctime()}"
repo.index.commit(commit_message)

print("Attempting to push to main branch.")
origin = repo.remote(name='origin')

try:
    origin.push(refspec='HEAD:main', force=True)
    print("Main branch updated successfully.")
except Exception as e:
    print(f"Push to main branch failed: {e}")


user = "user01"

# ======================================
# Commit History
#
# Instructions:
# This short code snippet displays the
# commit history of a repository file.
# ======================================

import datetime
from git import Repo

file_name = "AI_Modeler_LLM_Demo_01.py"
repo_dir = f"../{user}/repos"

repo = Repo(repo_dir)
commits = repo.iter_commits(paths=file_name)

print(f"History for {file_name}:")
for c in commits:
    date = datetime.datetime.fromtimestamp(c.committed_date).strftime('%Y-%m-%d %H:%M')
    print(f"{date} | {c.hexsha[:7]} | {c.message.strip()}")


user = "user01"

# ======================================
# Commit History
#
# Instructions:
# This short code snippet displays the
# commit history of a repository file.
# ======================================

import datetime
from git import Repo

file_name = "AI_Modeler_LLM_Demo_01.py"
repo_dir = f"../{user}/repos"

repo = Repo(repo_dir)
commits = repo.iter_commits(paths=file_name)

print(f"History for {file_name}:")
for c in commits:
    date = datetime.datetime.fromtimestamp(c.committed_date).strftime('%Y-%m-%d %H:%M')
    print(f"{date} | {c.hexsha[:7]} | {c.message.strip()}")


user = "user01"

# ======================================
# Commit History
#
# Instructions:
# This short code snippet displays the
# commit history of a repository file.
# ======================================

import datetime
from git import Repo

file_name = "AI_Modeler_LLM_Demo_02.py"
repo_dir = f"../{user}/repos"

repo = Repo(repo_dir)
commits = repo.iter_commits(paths=file_name)

print(f"History for {file_name}:")
for c in commits:
    date = datetime.datetime.fromtimestamp(c.committed_date).strftime('%Y-%m-%d %H:%M')
    print(f"{date} | {c.hexsha[:7]} | {c.message.strip()}")


import nbformat

input_path = "AI_Modeler_LLM_Demo_01.ipynb"    
output_path = "AI_Modeler_LLM_Demo_01.py"
cell_indexes = None # cell_indexes = [0, 1, 3]

def export_notebook():

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except FileNotFoundError:
        print(f"Notebook file not found: {input_path}")
        return
    except nbformat.reader.NotJSONError:
        print(f"Invalid notebook format: {input_path}")
        return

    code_snippets = []

    indexes_to_use = cell_indexes if cell_indexes is not None else range(len(nb.cells))

    for i in indexes_to_use:
        if i < len(nb.cells):
            cell = nb.cells[i]
            if cell.cell_type == 'code' and cell.source.strip():
                code_snippets.append(cell.source)
            elif cell.cell_type != 'code':
                print(f"Cell {i} is not a code cell.")
            else:
                print(f"Cell {i} is empty. Skipping.")
        else:
            print(f"Cell {i} is out of range.")

    if code_snippets:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(code_snippets))
        print(f"Code successfully saved to {output_path}")
    else:
        print("No code cells were exported.")

export_notebook()


import nbformat

input_path = "AI_Modeler_LLM_Demo_01.ipynb"    
output_path = "AI_Modeler_LLM_Demo_01.py"
cell_indexes = None # cell_indexes = [0, 1, 2]

def export_notebook():

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except FileNotFoundError:
        print(f"Notebook file not found: {input_path}")
        return
    except nbformat.reader.NotJSONError:
        print(f"Invalid notebook format: {input_path}")
        return

    code_snippets = []

    indexes_to_use = cell_indexes if cell_indexes is not None else range(len(nb.cells))

    for i in indexes_to_use:
        if i < len(nb.cells):
            cell = nb.cells[i]
            if cell.cell_type == 'code' and cell.source.strip():
                code_snippets.append(cell.source)
            elif cell.cell_type != 'code':
                print(f"Cell {i} is not a code cell.")
            else:
                print(f"Cell {i} is empty. Skipping.")
        else:
            print(f"Cell {i} is out of range.")

    if code_snippets:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(code_snippets))
        print(f"Code successfully saved to {output_path}")
    else:
        print("No code cells were exported.")

export_notebook()


import nbformat

input_path = "AI_Modeler_LLM_Demo_01.ipynb"    
output_path = "AI_Modeler_LLM_Demo_01.py"
cell_indexes = None # cell_indexes = [0, 1, 2]

def export_notebook():

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except FileNotFoundError:
        print(f"Notebook file not found: {input_path}")
        return
    except nbformat.reader.NotJSONError:
        print(f"Invalid notebook format: {input_path}")
        return

    code_snippets = []

    indexes_to_use = cell_indexes if cell_indexes is not None else range(len(nb.cells))

    for i in indexes_to_use:
        if i < len(nb.cells):
            cell = nb.cells[i]
            if cell.cell_type == 'code' and cell.source.strip():
                code_snippets.append(cell.source)
            elif cell.cell_type != 'code':
                print(f"Cell {i} is not a code cell.")
            else:
                print(f"Cell {i} is empty. Skipping.")
        else:
            print(f"Cell {i} is out of range.")

    if code_snippets:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(code_snippets))
        print(f"Code successfully saved to {output_path}")
    else:
        print("No code cells were exported.")

export_notebook()

# Main Sync: Mon Aug 31 20:03:50 2026
# Main Sync: Mon Aug 31 20:06:22 2026
# Main Sync: Mon Aug 31 20:10:08 2026