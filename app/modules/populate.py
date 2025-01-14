import traceback
import os
import shutil
import json
from app import create_app, logger, db
from app.models import *


def generate_syndrome_code(title, category, existing_codes):
    """
    Generate a unique syndrome code based on:
    - First letter of category (A-Z)
    - First 3 consonants from the syndrome name (uppercase)
    - 3 digit number sequence starting from 001
    """
    # Get first letter of category (use first letter of title if no category)
    cat_letter = title[0].upper()

    # Get consonants from title
    consonants = "".join(
        c.upper() for c in title if c.upper() in "BCDFGHJKLMNPQRSTVWXYZ"
    )
    if len(consonants) < 3:
        # Pad with X if not enough consonants
        consonants = (consonants + "XXX")[:3]
    else:
        consonants = consonants[:3]

    # Create base code
    base_code = f"{cat_letter}{consonants}"

    # Try numbers until we find an unused code
    counter = 1
    while True:
        code = f"{base_code}{counter:03d}"
        if code not in existing_codes:
            return code
        counter += 1


def process_directories(base_dir):
    """Process all syndrome directories to find composite images"""

    with open("syndromes.json") as file:
        data = json.load(file)
        updated_count = 0
        failed = 0
        for record in data:
            category = (
                str(record["directory"][:1]).upper()
                if str(record["directory"][:1]).isalpha()
                else "#"
            )
            syndrome_dir = os.path.join(base_dir, category, record["directory"])
            if not os.path.isdir(syndrome_dir):
                continue
            syndrome = Syndrome.query.filter(
                Syndrome.title == record["title"]
            ).one_or_none()
            if syndrome:
                logger.info(f"Found syndrome_dir: {syndrome_dir}")
                updated_count += 1
            else:
                failed += 1
                continue
            images = []
            composite = None
            for gender in ["man", "woman"]:
                gender_dir = os.path.join(syndrome_dir, gender)
                if not os.path.isdir(gender_dir):
                    continue
                for filename in os.listdir(gender_dir):
                    image_path = os.path.join(gender_dir, filename)
                    new_path = os.path.join(
                        app.static_folder,
                        "img",
                        "syndromes",
                        category,
                        syndrome.code,
                        f"{syndrome.code}_{len(images)}.jpg",
                    )
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    shutil.copy2(image_path, new_path)
                    logger.info(f"Copied: {filename} to {new_path}")
                    if "photo_1_" in filename:
                        composite = str(new_path).replace(f"{app.static_folder}/", "")
                    images.append(
                        {"path": str(new_path).replace(f"{app.static_folder}/", "")}
                    )
            # Update images
            syndrome.composite_image = composite
            syndrome.images = json.dumps(images)
            syndrome.update()

        logger.info(f"Updated: {updated_count}")
        logger.info(f"Failed: {failed}")


app = create_app()

with app.app_context():
    try:
        process_directories("preprocessed")
        # # Delete all exting syndromes
        # logger.info(f"Deleted: {Syndrome.query.delete()} syndromes")

        # # Set to store used codes
        # existing_codes = set()

        # # First pass: Generate codes for all records
        # with open("syndromes.json") as file:
        #     data = json.load(file)

        #     # Add codes to each record
        #     for record in data:
        #         code = generate_syndrome_code(
        #             record["title"],
        #             record.get("category", record["title"][0]),
        #             existing_codes,
        #         )
        #         record["code"] = code
        #         existing_codes.add(code)
        #         logger.info(f"Generated code {code} for {record['title']}")

        # # Second pass: Save to database with generated codes
        # for record in data:
        #     syndrome = Syndrome(
        #         title=record["title"],
        #         code=record["code"],  # Now we have the code
        #         synonyms="".join(record["synonyms"]),
        #         omim=record["omim"],
        #         genes=",".join(record["genes"]),
        #         location=record["location"],
        #         images=json.dumps([]),
        #         inheritance_modes=",".join(record["inheritance_modes"]),
        #         abstract=record["abstract"],
        #         features=",".join(record["features"]),
        #         resources=json.dumps(record["resources"]),
        #     )
        #     syndrome.insert()
        #     logger.info(f"Entered Syndrome #{syndrome.id} with code {syndrome.code}")

        # # Optionally save the updated JSON with codes
        # with open("syndromes_with_codes.json", "w") as f:
        #     json.dump(data, f, indent=2)

    except:
        logger.error(traceback.format_exc())
        db.session.rollback()
