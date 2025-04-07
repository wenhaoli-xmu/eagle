try:
    import transformer_engine
    import transformer_engine_extensions
except:
    print("having trouble importing transformer-engine!")
    
from train_v4 import train

if __name__ == "__main__":
    train()

