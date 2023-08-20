#!/bin/bash

# Change to the specified directory
cd /home/ziad/GitHub/Machine-Learning-journey || { echo "Failed to change directory"; exit 1; }

# Check if there are any changes to commit
if [ -n "$(git status --porcelain)" ]; then
  # Add all changes to the staging area
  git add *

  # Prompt the user for a commit message
  echo "Enter a commit message: "
  read -r commit_message

  # Commit the changes with the specified message
  git commit -m "$commit_message"

  # Push the changes to the remote repository
  git push origin main
else
  echo "No changes to commit"
fi

