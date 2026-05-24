"""Demo: DSPy optimizer for iterative learning and improvement."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drg.optimizer import create_optimizer
from drg.schema import DRGSchema, Entity, Relation


def main():
    """Run optimizer demo."""

    # Set Gemini as default model
    if not os.getenv("DRG_MODEL"):
        os.environ["DRG_MODEL"] = "gemini/gemini-2.0-flash-exp"

    print("=" * 60)
    print("DRG Optimizer Demo: Iterative Learning")
    print("=" * 60)

    # Define schema
    schema = DRGSchema(
        entities=[
            Entity("Company"),
            Entity("Person"),
            Entity("Product"),
            Entity("Location"),
        ],
        relations=[
            Relation("founded_by", "Company", "Person"),
            Relation("produces", "Company", "Product"),
            Relation("located_in", "Company", "Location"),
            Relation("ceo_of", "Person", "Company"),
        ],
    )

    # Training examples
    print("\n1. Preparing training examples...")
    training_examples = [
        {
            "text": "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
            "expected_entities": [
                ("Apple Inc.", "Company"),
                ("Steve Jobs", "Person"),
                ("Steve Wozniak", "Person"),
                ("Ronald Wayne", "Person"),
            ],
            "expected_relations": [
                ("Apple Inc.", "founded_by", "Steve Jobs"),
                ("Apple Inc.", "founded_by", "Steve Wozniak"),
                ("Apple Inc.", "founded_by", "Ronald Wayne"),
            ],
        },
        {
            "text": "Microsoft produces Windows and Office software. Bill Gates is the CEO of Microsoft.",
            "expected_entities": [
                ("Microsoft", "Company"),
                ("Windows", "Product"),
                ("Office", "Product"),
                ("Bill Gates", "Person"),
            ],
            "expected_relations": [
                ("Microsoft", "produces", "Windows"),
                ("Microsoft", "produces", "Office"),
                ("Bill Gates", "ceo_of", "Microsoft"),
            ],
        },
        {
            "text": "Google is located in Mountain View, California. The company produces Android and Chrome.",
            "expected_entities": [
                ("Google", "Company"),
                ("Mountain View", "Location"),
                ("California", "Location"),
                ("Android", "Product"),
                ("Chrome", "Product"),
            ],
            "expected_relations": [
                ("Google", "located_in", "Mountain View"),
                ("Google", "located_in", "California"),
                ("Google", "produces", "Android"),
                ("Google", "produces", "Chrome"),
            ],
        },
    ]

    print(f"   Prepared {len(training_examples)} training examples")

    # Test examples
    print("\n2. Preparing test examples...")
    test_examples = [
        {
            "text": "Tesla was founded by Elon Musk. The company produces electric vehicles and is located in Austin, Texas.",
            "expected_entities": [
                ("Tesla", "Company"),
                ("Elon Musk", "Person"),
                ("electric vehicles", "Product"),
                ("Austin", "Location"),
                ("Texas", "Location"),
            ],
            "expected_relations": [
                ("Tesla", "founded_by", "Elon Musk"),
                ("Tesla", "produces", "electric vehicles"),
                ("Tesla", "located_in", "Austin"),
                ("Tesla", "located_in", "Texas"),
            ],
        },
        {
            "text": "Amazon produces AWS cloud services. Jeff Bezos was the CEO of Amazon.",
            "expected_entities": [
                ("Amazon", "Company"),
                ("AWS", "Product"),
                ("Jeff Bezos", "Person"),
            ],
            "expected_relations": [
                ("Amazon", "produces", "AWS"),
                ("Jeff Bezos", "ceo_of", "Amazon"),
            ],
        },
    ]

    print(f"   Prepared {len(test_examples)} test examples")

    # Create optimizer
    print("\n3. Creating optimizer...")
    optimizer = create_optimizer(
        schema=schema,
        optimizer_type="bootstrap_few_shot",
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
    )

    # Add training examples
    for example in training_examples:
        optimizer.add_training_example(
            text=example["text"],
            expected_entities=example["expected_entities"],
            expected_relations=example["expected_relations"],
        )

    print("   Optimizer created with BootstrapFewShot strategy")

    # Evaluate baseline (before optimization)
    print("\n4. Evaluating baseline (before optimization)...")
    try:
        baseline_result = optimizer.evaluate(test_examples, use_optimized=False)
        print(f"   Baseline F1: {baseline_result.f1:.3f}")
        print(f"   Baseline Precision: {baseline_result.precision:.3f}")
        print(f"   Baseline Recall: {baseline_result.recall:.3f}")
    except Exception as e:
        print(f"   Baseline evaluation failed ({type(e).__name__}), using mock metrics...")
        # Mock baseline for demo
        from drg.optimizer.optimizer import EvaluationResult

        baseline_result = EvaluationResult(
            precision=0.65,
            recall=0.60,
            f1=0.62,
            accuracy=0.58,
            details={},
        )
        print(f"   Mock Baseline F1: {baseline_result.f1:.3f}")

    # Optimize
    print("\n5. Running optimization...")
    try:
        optimizer.optimize()
        print("   Optimization completed")
    except Exception as e:
        print(f"   Optimization failed ({type(e).__name__}), using base extractor...")

    # Evaluate optimized (after optimization)
    print("\n6. Evaluating optimized extractor...")
    try:
        optimized_result = optimizer.evaluate(test_examples, use_optimized=True)
        print(f"   Optimized F1: {optimized_result.f1:.3f}")
        print(f"   Optimized Precision: {optimized_result.precision:.3f}")
        print(f"   Optimized Recall: {optimized_result.recall:.3f}")
    except Exception as e:
        print(f"   Optimized evaluation failed ({type(e).__name__}), using mock metrics...")
        # Mock optimized for demo
        from drg.optimizer.optimizer import EvaluationResult

        optimized_result = EvaluationResult(
            precision=0.75,
            recall=0.72,
            f1=0.73,
            accuracy=0.70,
            details={},
        )
        print(f"   Mock Optimized F1: {optimized_result.f1:.3f}")

    # Compare before/after
    print("\n7. Before/After Comparison:")
    print("-" * 60)
    comparison = optimizer.compare_before_after(test_examples)

    print("\n   Precision:")
    print(f"     Before: {comparison['base']['precision']:.3f}")
    print(f"     After:  {comparison['optimized']['precision']:.3f}")
    print(
        f"     Improvement: {comparison['improvement']['precision']:+.3f} ({comparison['improvement_percent']['precision']:+.1f}%)"
    )

    print("\n   Recall:")
    print(f"     Before: {comparison['base']['recall']:.3f}")
    print(f"     After:  {comparison['optimized']['recall']:.3f}")
    print(
        f"     Improvement: {comparison['improvement']['recall']:+.3f} ({comparison['improvement_percent']['recall']:+.1f}%)"
    )

    print("\n   F1 Score:")
    print(f"     Before: {comparison['base']['f1']:.3f}")
    print(f"     After:  {comparison['optimized']['f1']:.3f}")
    print(
        f"     Improvement: {comparison['improvement']['f1']:+.3f} ({comparison['improvement_percent']['f1']:+.1f}%)"
    )

    # Iterative improvement
    print("\n8. Iterative Improvement Loop:")
    print("-" * 60)
    try:
        iteration_results = optimizer.iterative_improve(
            test_examples=test_examples,
            max_iterations=3,
        )

        print("\n   Iteration Results:")
        for i, result in enumerate(iteration_results, 1):
            print(
                f"     Iteration {i}: F1={result.f1:.3f}, Precision={result.precision:.3f}, Recall={result.recall:.3f}"
            )
    except Exception as e:
        print(f"   Iterative improvement failed ({type(e).__name__})")
        print("   (This requires working LLM API)")

    print("\n" + "=" * 60)
    print("Optimizer demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
