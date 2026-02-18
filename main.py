from inference.paraphrase import ParaphraseSystem


def main():
    with open("Data/test_passage.txt", "r", encoding="utf-8") as f:
        text = f.read()

    system = ParaphraseSystem()
    system.run_comparison(text, min_length_ratio=0.8)


if __name__ == "__main__":
    main()
