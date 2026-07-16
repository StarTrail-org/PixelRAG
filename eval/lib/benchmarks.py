"""
Dataset loading functions for visual/multimodal QA benchmarks.

Standalone dataset loaders for the eval pipeline (SimpleQA, NQ, NQ-Tables, EVQA,
MMSearch, WorldVQA, ...), with no external dependencies.
"""

import base64
import hashlib
import io
import json
import logging
import random
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import datasets
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_TASKS = {
    "2wiki": "akariasai/2wiki_rand1k",
    "worldvqa": "moonshotai/WorldVQA",
    "simplevqa": "m-a-p/SimpleVQA",
    "factualvqa": "lmms-lab/FVQA",
    "mmsearch": "CaraJ/MMSearch",
    "webqa": "Anil99/webqa",
    "multimodalqa": "allenai/multimodalqa",
}

# img_id with 404 URL (Verizonnyc.jpg); examples with ONLY this img_id have no fallback
EVQA_LANDMARK_404_IMG_ID = "160a34689b4542f2"

# Example IDs where all img_id URLs are 404 (no fallback); skip when loading
EVQA_LANDMARK_SKIP_IDS = frozenset(
    {
        "e87957e51e4606ab56d5f475e80fc353",  # all 5 URLs 404 (question: temple hidden structure, Shanxi Taiyuan historic sites series)
        "62e1cbe1009909d6ff448063c6308719",  # all 5 URLs 404 (question: Monument to the Conquerors of Space coin year, 2_hop)
    }
)

DATASET_URLS = {
    "encyclopedic_vqa_val": "https://storage.googleapis.com/encyclopedic-vqa/val.csv",
    "encyclopedic_vqa_test": "https://storage.googleapis.com/encyclopedic-vqa/test.csv",
}


def get_cache_dir() -> Path:
    """Get the cache directory for downloaded datasets."""
    cache_dir = Path.home() / ".cache" / "pixelrag_eval" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(url: str, cache_name: str) -> Path:
    """Download file from URL to cache directory if not already cached."""
    cache_path = get_cache_dir() / cache_name
    if not cache_path.exists():
        urllib.request.urlretrieve(url, cache_path)
    return cache_path


def _bytes_to_pil(raw_bytes) -> Optional[Image.Image]:
    """Convert raw bytes, base64 string, dict with 'bytes' key, or PIL Image to a PIL Image.
    Returns None on failure."""
    try:
        if isinstance(raw_bytes, dict) and "bytes" in raw_bytes:
            raw_bytes = raw_bytes["bytes"]
        if isinstance(raw_bytes, list):
            raw_bytes = bytes(raw_bytes)
        if isinstance(raw_bytes, str):
            # Try base64 decoding
            try:
                decoded = base64.b64decode(raw_bytes)
                return Image.open(io.BytesIO(decoded)).convert("RGB")
            except Exception:
                pass
            return None
        if isinstance(raw_bytes, (bytes, bytearray)):
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        # Already a PIL Image (HuggingFace datasets sometimes auto-decode)
        if isinstance(raw_bytes, Image.Image):
            return raw_bytes.convert("RGB")
    except Exception as e:
        logger.debug(f"Failed to convert bytes to PIL Image: {e}")
    return None


def load_encyclopedic_vqa_data(
    split: str = "val",
    num_examples: Optional[int] = None,
    shuffle: bool = False,
    local_path: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    question_type_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Load Encyclopedic VQA dataset.

    Args:
        split: Dataset split ('val' or 'test')
        num_examples: Limit to first N examples (optional)
        shuffle: Whether to shuffle the examples
        local_path: Optional local path to dataset CSV
        dataset_filter: Filter by dataset_name ('inaturalist' or 'landmarks')
        question_type_filter: Filter by question_type ('templated', 'automatic', 'multi_answer', '2_hop')

    Returns:
        List of Encyclopedic VQA examples
    """
    if local_path and Path(local_path).exists():
        df = pd.read_csv(local_path)
    else:
        url_key = f"encyclopedic_vqa_{split}"
        cache_name = f"encyclopedic_vqa_{split}.csv"
        cache_path = download_file(DATASET_URLS[url_key], cache_name)
        df = pd.read_csv(cache_path)

    examples = []
    for idx, row in df.iterrows():
        question = str(row.get("question", ""))
        answer_raw = str(row.get("answer", ""))
        # Answers are pipe-separated
        reference_list = [a.strip() for a in answer_raw.split("|") if a.strip()]

        # Use question + wikipedia_url + row index for ID to avoid collisions
        # (templated questions repeat across species, and same species can have multiple image sets)
        wiki_url = str(row.get("wikipedia_url", ""))
        id_source = f"{question}|{wiki_url}|{idx}"
        example = {
            "id": hashlib.md5(id_source.encode()).hexdigest(),
            "problem": question,
            "answer": answer_raw,
            "reference_list": reference_list,
            "question_type": str(row.get("question_type", "automatic")),
            "additional_instructions": (
                "Your final response should be in the following format:\n"
                "Exact Answer: <your succinct, final answer>"
            ),
        }
        # Preserve optional metadata columns
        for col in [
            "wikipedia_url",
            "wikipedia_title",
            "question_original",
            "dataset_image_ids",
            "dataset_name",
            "wikipedia_url_used_in_train",
        ]:
            if col in row.index and pd.notna(row[col]):
                example[col] = row[col]

        # Map wikipedia_url into metadata so screenshot/retrieval pipeline can find it
        if "wikipedia_url" in example and example["wikipedia_url"]:
            example["metadata"] = {"url": example["wikipedia_url"]}

        # Parse dataset_image_ids for query images (iNaturalist or Google Landmarks)
        if "dataset_image_ids" in example and example["dataset_image_ids"]:
            raw_ids = str(example["dataset_image_ids"])
            ids = [i.strip() for i in raw_ids.split("|") if i.strip()]
            example["dataset_image_ids_parsed"] = ids
            if example.get("dataset_name", "").lower() == "inaturalist":
                example["inat_image_ids"] = ids  # backward compat

        examples.append(example)

    if dataset_filter:
        ds_lower = dataset_filter.lower()
        examples = [
            e for e in examples if (e.get("dataset_name") or "").lower() == ds_lower
        ]

    if question_type_filter:
        allowed_qts = frozenset(
            q.strip().lower() for q in question_type_filter.split(",") if q.strip()
        )
        examples = [
            e for e in examples if (e.get("question_type") or "").lower() in allowed_qts
        ]

    # Skip landmark examples with 404 query image URLs
    if dataset_filter and (dataset_filter.lower() == "landmarks"):

        def _has_404_only(e):
            ids = e.get("dataset_image_ids_parsed") or []
            return ids and set(ids) == {EVQA_LANDMARK_404_IMG_ID}

        examples = [
            e
            for e in examples
            if e.get("id") not in EVQA_LANDMARK_SKIP_IDS and not _has_404_only(e)
        ]

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples


def load_shortformqa_data(
    dataset_repo: str, num_examples: Optional[int] = None, shuffle: bool = False
) -> List[Dict]:
    """
    Load Short-form QA dataset data.

    Args:
        dataset_repo: HuggingFace dataset repository name
        num_examples: Limit to first N examples (optional)
        shuffle: Whether to shuffle the examples

    Returns:
        List of Short-form QA examples
    """
    dataset = datasets.load_dataset(dataset_repo, split="test")
    examples = []
    for example in dataset:
        example["problem"] = example["messages"][-1]["content"]
        example["id"] = hashlib.md5(example["problem"].encode()).hexdigest()
        example["answers"] = (
            json.loads(example["ground_truth"])
            if example["ground_truth"][0] == "["
            else [example["ground_truth"]]
        )
        example["additional_instructions"] = """
Your final response should be in the following format without any other text:
Exact Answer: <your succinct, final answer>
""".strip()
        examples.append(example)

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples


def load_worldvqa_data(
    num_examples: Optional[int] = None,
    shuffle: bool = False,
    language_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Load WorldVQA dataset from HuggingFace.

    Args:
        num_examples: Limit to first N examples (optional)
        shuffle: Whether to shuffle the examples

    Returns:
        List of WorldVQA examples
    """
    dataset = datasets.load_dataset("moonshotai/WorldVQA", split="train")

    examples = []
    for idx, sample in enumerate(dataset):
        lang = sample.get("language", "")
        # Filter out Chinese examples by default
        if lang == "zh":
            continue

        example = {
            "id": str(idx),
            "problem": sample["question"],
            "answer": sample["answer"],
            "additional_instructions": (
                "Your final response should be in the following format:\n"
                "Exact Answer: <your succinct, final answer>"
            ),
        }
        # Preserve metadata
        for col in ["image", "category", "difficulty", "language"]:
            if col in sample:
                example[col] = sample[col]

        examples.append(example)

    if language_filter:
        examples = [ex for ex in examples if ex.get("language") == language_filter]

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples


def load_simplevqa_data(
    num_examples: Optional[int] = None,
    shuffle: bool = False,
) -> List[Dict]:
    """
    Load SimpleVQA dataset (m-a-p/SimpleVQA, test split, ~2030 examples).
    Multi-modal factual VQA benchmark with images.

    Columns: data_id, image, image_description, language, question, answer,
             original_category, source, atomic_question, atomic_fact, vqa_category.

    Filters out Chinese-language examples by default.

    Returns:
        List of dicts with keys: id, problem, answer, image (PIL), additional_instructions, + metadata.
    """
    dataset = datasets.load_dataset("m-a-p/SimpleVQA", split="test")

    examples = []
    for sample in dataset:
        lang = sample.get("language", "")
        # Filter out Chinese examples
        if lang and lang.lower() in ("chinese", "zh", "cn"):
            continue

        pil_image = None
        raw_img = sample.get("image")
        if raw_img is not None:
            pil_image = _bytes_to_pil(raw_img)

        example = {
            "id": str(sample["data_id"]),
            "problem": sample["question"],
            "answer": sample["answer"],
            "image": pil_image,
            "additional_instructions": (
                "Your final response should be in the following format:\n"
                "Exact Answer: <your succinct, final answer>"
            ),
            # Metadata
            "language": lang,
            "original_category": sample.get("original_category", ""),
            "vqa_category": sample.get("vqa_category", ""),
            "source": sample.get("source", ""),
        }
        examples.append(example)

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples


def load_factualvqa_data(
    num_examples: Optional[int] = None,
    shuffle: bool = False,
) -> List[Dict]:
    """
    Load FactualVQA dataset (lmms-lab/FVQA, train split).
    Factual VQA benchmark with search-required / search-free annotations.

    Columns: data_id, images (list of image dicts), prompt (list of message dicts),
             reward_model (dict with ground_truth), category (search_required/search_free).

    Returns:
        List of dicts with keys: id, problem, answer, image (PIL), additional_instructions, + metadata.
    """
    dataset = datasets.load_dataset("lmms-lab/FVQA", split="train")

    examples = []
    for sample in dataset:
        # Extract question from prompt[0]["content"]
        prompt_list = sample.get("prompt", [])
        if not prompt_list:
            continue
        question = prompt_list[0].get("content", "")
        if not question:
            continue

        # Extract answer from reward_model["ground_truth"]
        reward_model = sample.get("reward_model", {})
        if isinstance(reward_model, str):
            try:
                reward_model = json.loads(reward_model)
            except (json.JSONDecodeError, TypeError):
                reward_model = {}
        answer = reward_model.get("ground_truth", "")
        if not answer:
            continue

        # Extract first image
        pil_image = None
        images_list = sample.get("images", [])
        if images_list:
            pil_image = _bytes_to_pil(images_list[0])

        example = {
            "id": str(
                sample.get("data_id", hashlib.md5(question.encode()).hexdigest())
            ),
            "problem": question,
            "answer": answer,
            "image": pil_image,
            "additional_instructions": (
                "Your final response should be in the following format:\n"
                "Exact Answer: <your succinct, final answer>"
            ),
            # Metadata
            "category": sample.get("category", ""),
            "data_source": sample.get("data_source", ""),
        }
        examples.append(example)

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples


def load_mmsearch_data(
    num_examples: Optional[int] = None,
    shuffle: bool = False,
) -> List[Dict]:
    """
    Load MMSearch dataset (CaraJ/MMSearch, end2end config, 300 examples).
    Multimodal search benchmark with text queries, query images, and ground-truth answers.

    Columns: sample_id, query, query_image, image_search_result, area, subfield,
             timestamp, gt_requery, gt_answer, alternative_gt_answers.

    Returns:
        List of dicts with keys: id, problem, answer, image (PIL), additional_instructions, + metadata.
    """
    dataset = datasets.load_dataset("CaraJ/MMSearch", "end2end", split="end2end")

    examples = []
    for sample in dataset:
        pil_image = None
        raw_img = sample.get("query_image")
        if raw_img is not None:
            pil_image = _bytes_to_pil(raw_img)

        alt_answers = sample.get("alternative_gt_answers", [])
        if isinstance(alt_answers, str):
            try:
                alt_answers = json.loads(alt_answers)
            except (json.JSONDecodeError, TypeError):
                alt_answers = [alt_answers] if alt_answers else []

        gt_answer = sample.get("gt_answer", "")
        # Build combined answer string for evaluation: primary + alternatives
        all_answers = [gt_answer] + [a for a in alt_answers if a]
        answer_str = " | ".join(all_answers) if len(all_answers) > 1 else gt_answer

        example = {
            "id": str(sample.get("sample_id", "")),
            "problem": sample.get("query", ""),
            "answer": answer_str,
            "image": pil_image,
            "additional_instructions": (
                "Your final response should be in the following format:\n"
                "Exact Answer: <your succinct, final answer>"
            ),
            # Metadata
            "alternative_gt_answers": alt_answers,
            "gt_answer": gt_answer,
            "area": sample.get("area", ""),
            "subfield": sample.get("subfield", ""),
            "timestamp": sample.get("timestamp", ""),
            "gt_requery": sample.get("gt_requery", ""),
        }
        examples.append(example)

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples


def load_webqa_data(
    num_examples: Optional[int] = None,
    shuffle: bool = False,
) -> List[Dict]:
    """
    Load WebQA dataset (Anil99/webqa, validation split).
    Multimodal multi-hop reasoning benchmark where each question has text and/or image sources.

    NOTE: This dataset is large and may be slow to load. The HuggingFace viewer cannot
    render it due to row size (>1.4MB per row). We load the validation split and extract
    the question, answer, and image (if available) from the source snippets.

    If loading fails (e.g. dataset is gated, too large, or schema mismatch),
    this function logs a warning and returns an empty list.

    Returns:
        List of dicts with keys: id, problem, answer, image (PIL or None), additional_instructions.
    """
    try:
        # Use streaming to avoid memory issues with large rows (>1.4MB each)
        dataset = datasets.load_dataset(
            "Anil99/webqa", split="validation", streaming=True
        )
    except Exception as e:
        logger.warning(
            f"Failed to load WebQA dataset (Anil99/webqa): {e}. "
            "This dataset may require special handling due to its large row sizes. "
            "Returning empty list."
        )
        return []

    examples = []
    for idx, sample in enumerate(dataset):
        # WebQA structure varies; try common field names
        question = sample.get("question", sample.get("Q", ""))
        if not question:
            continue

        answer = sample.get("answer", sample.get("A", ""))
        if not answer:
            # Try extracting from Qcate or other fields
            answer = str(sample.get("answer", ""))

        # Try to extract an image from the sample
        pil_image = None
        # WebQA stores images in positive/negative fact lists; try to get one
        for img_key in ["img_posFacts", "img_pos", "image", "images"]:
            img_data = sample.get(img_key)
            if img_data is not None:
                if isinstance(img_data, list) and len(img_data) > 0:
                    first_item = img_data[0]
                    if isinstance(first_item, dict):
                        raw = first_item.get("image", first_item.get("bytes"))
                        if raw is not None:
                            pil_image = _bytes_to_pil(raw)
                    else:
                        pil_image = _bytes_to_pil(first_item)
                else:
                    pil_image = _bytes_to_pil(img_data)
                if pil_image is not None:
                    break

        example = {
            "id": str(sample.get("id", sample.get("guid", idx))),
            "problem": question,
            "answer": str(answer),
            "image": pil_image,
            "additional_instructions": (
                "Your final response should be in the following format:\n"
                "Exact Answer: <your succinct, final answer>"
            ),
            # Metadata
            "Qcate": sample.get("Qcate", ""),
        }
        examples.append(example)

        # With streaming, stop early once we have enough
        if num_examples and not shuffle and len(examples) >= num_examples:
            break

    if shuffle:
        random.seed(42)
        random.shuffle(examples)
        if num_examples:
            examples = examples[:num_examples]

    return examples


def load_multimodalqa_data(
    num_examples: Optional[int] = None,
    shuffle: bool = False,
) -> List[Dict]:
    """
    Load MultiModalQA dataset (allenai/multimodalqa).
    Cross-modal QA benchmark requiring reasoning over text, tables, and images.

    NOTE: This dataset is hosted on GitHub (not HuggingFace). Images require a
    separate 3.6GB download from S3 (images.zip). This loader attempts to load
    the dev split questions from HuggingFace (community mirror) or falls back to
    downloading from the official GitHub release. Images are NOT loaded automatically;
    the `image` field will be None unless the images are pre-downloaded to
    ~/.cache/pixelrag_eval/datasets/multimodalqa_images/.

    If no HuggingFace mirror is available, we download the dev JSONL directly from GitHub.

    Returns:
        List of dicts with keys: id, problem, answer, image (PIL or None), additional_instructions, + metadata.
    """
    import gzip

    cache_dir = get_cache_dir()
    dev_jsonl_path = cache_dir / "MultiModalQA_dev.jsonl"

    # Try loading from HuggingFace mirror first, fall back to GitHub raw files
    questions = []
    try:
        # Try the official GitHub raw file
        if not dev_jsonl_path.exists():
            dev_gz_url = "https://raw.githubusercontent.com/allenai/multimodalqa/master/dataset/MMQA_dev.jsonl.gz"
            gz_path = cache_dir / "MultiModalQA_dev.jsonl.gz"
            logger.info("Downloading MultiModalQA dev set from GitHub...")
            urllib.request.urlretrieve(dev_gz_url, gz_path)
            with gzip.open(gz_path, "rt", encoding="utf-8") as f_in:
                with open(dev_jsonl_path, "w", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())
            gz_path.unlink(missing_ok=True)

        with open(dev_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
    except Exception as e:
        logger.warning(
            f"Failed to load MultiModalQA dataset: {e}. "
            "The dataset requires downloading from GitHub "
            "(https://github.com/allenai/multimodalqa). Returning empty list."
        )
        return []

    if not questions:
        logger.warning("MultiModalQA dev set is empty after loading.")
        return []

    # Check if images directory exists for optional image loading
    # Images may be in multimodalqa_images/ or multimodalqa_images/final_dataset_images/
    images_dir = cache_dir / "multimodalqa_images" / "final_dataset_images"
    if not images_dir.is_dir():
        images_dir = cache_dir / "multimodalqa_images"
    has_images = images_dir.is_dir()
    if not has_images:
        logger.info(
            "MultiModalQA images not found at %s. Image field will be None. "
            "To enable images, download and extract: "
            "https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip "
            "into %s",
            images_dir,
            images_dir,
        )

    examples = []
    for sample in questions:
        qid = sample.get("qid", "")
        question_text = sample.get("question", "")
        if not question_text:
            continue

        # Extract answers (list of answer dicts)
        answers_raw = sample.get("answers", [])
        if isinstance(answers_raw, list):
            answer_texts = []
            for ans in answers_raw:
                if isinstance(ans, dict):
                    answer_texts.append(ans.get("answer", ""))
                elif isinstance(ans, str):
                    answer_texts.append(ans)
            answer_str = " | ".join(str(a) for a in answer_texts if a) or ""
        elif isinstance(answers_raw, str):
            answer_str = answers_raw
        else:
            answer_str = str(answers_raw)

        # Try to load image if images are downloaded
        pil_image = None
        if has_images:
            # MultiModalQA references images via metadata.image_doc_ids
            metadata = sample.get("metadata", {})
            image_doc_ids = metadata.get("image_doc_ids", [])
            for img_id in image_doc_ids:
                # Images are stored as {img_id}.jpg or {img_id}.png
                for ext in (".jpg", ".jpeg", ".png"):
                    img_path = images_dir / f"{img_id}{ext}"
                    if img_path.exists():
                        try:
                            pil_image = Image.open(img_path).convert("RGB")
                        except Exception:
                            pass
                        break
                if pil_image is not None:
                    break

        # Extract modality info
        metadata = sample.get("metadata", {})
        example = {
            "id": str(qid),
            "problem": question_text,
            "answer": answer_str,
            "image": pil_image,
            "additional_instructions": (
                "Your final response should be in the following format:\n"
                "Exact Answer: <your succinct, final answer>"
            ),
            # Metadata
            "reasoning_type": metadata.get("type", ""),
            "modalities": metadata.get("modalities", []),
        }
        examples.append(example)

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples
