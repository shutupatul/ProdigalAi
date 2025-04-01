# import matplotlib.pyplot as plt
# import re

# # Initialize lists to store extracted values
# iterations = []
# loss_values = []

# # Open and read the log file
# with open("gpt_log.txt", "r") as file:
#     for line in file:
#         # Match "Iteration X, Loss: Y"
#         match_iter = re.search(r"Iteration (\d+), Loss: ([\d.]+)", line)
        
#         if match_iter:
#             iteration = int(match_iter.group(1))
#             loss = float(match_iter.group(2))
#             iterations.append(iteration)
#             loss_values.append(loss)

# # Check if we have extracted any data
# if not iterations or not loss_values:
#     print("No loss data found. Check the log format.")
# else:
#     # Plot the loss curve
#     plt.figure(figsize=(8, 5))
#     plt.plot(iterations, loss_values, marker="o", linestyle="-", label="Loss")
    
#     plt.xlabel("Iteration")
#     plt.ylabel("Loss")
#     plt.title("Model Loss over Iterations")
#     plt.legend()
#     plt.grid()
#     plt.show()


import matplotlib.pyplot as plt
import re

# Initialize lists for step, train loss, and val loss
steps = []
train_losses = []
val_losses = []

# Open and read the log file
with open("bigram_log.txt", "r", encoding="utf-8") as file:

    for line in file:
        # Match "step X: train loss Y, val loss Z"
        match = re.search(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)", line)
        
        if match:
            step = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            
            steps.append(step)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

# Check if data was extracted
if not steps or not train_losses or not val_losses:
    print("No loss data found. Check the log format.")
else:
    # Plot the loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_losses, marker="o", linestyle="-", label="Train Loss", color="blue")
    plt.plot(steps, val_losses, marker="s", linestyle="--", label="Validation Loss", color="red")
    
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss over Steps")
    plt.legend()
    plt.grid()
    plt.show()