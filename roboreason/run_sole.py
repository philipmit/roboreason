import multiprocessing as mp

def main():
    from sole import load_model
    load_model()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

