import sys
from summarizer.pipeline import summarize_long
from summarizer.preprocessing import chunk_text, clean_text

def run_test():
    # Generate a long dummy document (approx 2000 words)
    paragraph = "The core purpose of this experiment is to validate the recursive long-form chunking logic of the text summarization platform. In numerous prior executions, the text would often lose meaning over successive chunk iterations, causing significant repetition or gibberish to appear in the ultimate summary output. By evaluating a carefully structured long text, we ensure that the overlap parameter prevents critical context from being dropped at chunk boundaries. "
    long_text = paragraph * 80 # ~3200 words

    with open("test_longform_out.txt", "w", encoding="utf-8") as f:
        f.write(f"Original Text length: {len(long_text.split())} words\n")
        
        # 1. Test clean_text on punctuation orphans
        test_clean = clean_text("This is an orphan . And another , see? Yes !")
        f.write(f"[Cleaner Test]: {test_clean}\n")
        
        # 2. Run the big text through the pipeline using bart-cnn
        f.write("\nRunning summarize_long with bart-cnn...\n")
        try:
            final_summary = summarize_long(long_text, "bart-cnn")
            f.write(f"\n[Final Summary length: {len(final_summary.split())} words]\n")
            f.write("-" * 50 + "\n")
            f.write(final_summary + "\n")
            f.write("-" * 50 + "\n")
        except Exception as e:
            f.write(f"Error during summarize_long: {e}\n")

if __name__ == "__main__":
    run_test()
