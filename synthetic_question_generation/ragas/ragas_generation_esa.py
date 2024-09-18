import json
import random
from tqdm import tqdm
import argparse
from ragas.llms.huggingface import HuggingfaceLLM
from ragas.embeddings.base import HuggingfaceEmbeddings
from ragas.testset import TestsetGenerator
from llama_index import Document

def main(input_path, output_path, test_size):
    # Load and shuffle data
    with open(input_path, 'r') as f:
        data = json.load(f)
    random.shuffle(data)

    # Initialize models
    model = HuggingfaceLLM("openchat/openchat_3.5")
    Embeddings = HuggingfaceEmbeddings()

    # Initialize TestsetGenerator
    testsetgenerator = TestsetGenerator(
        generator_llm=model,
        critic_llm=model,
        embeddings_model=Embeddings,
        testset_distribution={"easy": 1, "reasoning": 0.0, "conversation": 0.0},
        chat_qa=0.0,
        chunk_size=768,
        seed=42,
    )

    # Prepare documents
    documents = [Document(metadata={"document": item['document']}, text=item['text']) for item in data]

    # Generate testset
    testset = testsetgenerator.generate(documents, test_size=test_size)

    # Save testset
    with open(output_path, "w") as f:
        json.dump(testset, f, indent=4)

    print(f"Testset generated and saved to {output_path}")

if __name__ == "__main__":
    import json
import random
from tqdm import tqdm
import argparse
from ragas.llms.huggingface import HuggingfaceLLM
from ragas.embeddings.base import HuggingfaceEmbeddings
from ragas.testset import TestsetGenerator
from llama_index import Document

def main(args):
    # Load and shuffle data
    with open(args.input_path, 'r') as f:
        data = json.load(f)
    
    # Initialize models
    model = HuggingfaceLLM(args.model_name)
    Embeddings = HuggingfaceEmbeddings()

    # Initialize TestsetGenerator
    testsetgenerator = TestsetGenerator(
        generator_llm=model,
        critic_llm=model,
        embeddings_model=Embeddings,
        testset_distribution={"easy": args.easy, "reasoning": args.reasoning, "conversation": args.conversation},
        chat_qa=args.chat_qa,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    # Prepare documents
    documents = [Document(metadata={"document": item['document']}, text=item['text']) for item in data]

    # Shuffle documents
    random.shuffle(documents)

    # Generate testset
    testset = testsetgenerator.generate(documents, test_size=args.test_size)

    print(len(testset))
    # Save testset
    with open(args.output_path, "w") as f:
        json.dump(testset, f, indent=4)

    print(f"Testset generated and saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a testset using RAGAS TestsetGenerator")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSON file")
    parser.add_argument("--test_size", type=int, default=100, help="Number of test samples to generate")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name of the HuggingFace model to use")
    parser.add_argument("--easy", type=float, default=1.0, help="Proportion of easy questions")
    parser.add_argument("--reasoning", type=float, default=0.0, help="Proportion of reasoning questions")
    parser.add_argument("--conversation", type=float, default=0.0, help="Proportion of conversation questions")
    parser.add_argument("--chat_qa", type=float, default=0.0, help="Proportion of chat QA")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for text processing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set the random seed
    random.seed(args.seed)

    main(args)
