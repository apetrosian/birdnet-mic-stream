import os

import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
service_account_path = os.environ.get(
    "FIREBASE_SERVICE_ACCOUNT_PATH", "firebase_service_account.json"
)

cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred)

# Initialize Firestore DB
client = firestore.client()


class FirebaseExporter:
    def send(self, detection_data: list[dict]) -> None:

        if not client:
            raise ValueError("Firebase client not initialized.")

        try:
            # Ensure detection_data is a list
            if not isinstance(detection_data, list):
                detection_data = [detection_data]

            # Track highest confidence detection for each species
            best_detections = {}

            for detection in detection_data:
                # Create a unique key from common_name only
                common_name = detection.get("common_name")
                species_key = common_name
                confidence = detection.get("confidence", 0.0)

                # If species not seen before, or this detection has higher confidence
                if (
                    species_key not in best_detections
                    or confidence > best_detections[species_key]["confidence"]
                ):
                    best_detections[species_key] = detection

            # Create a batch write for best detections only
            batch = client.batch()
            unique_detections = list(best_detections.values())

            for detection in unique_detections:
                # Create a document reference with timestamp as ID
                doc_ref = client.collection("detections").document()

                # Prepare document data (no scientific_name)
                doc_data = {
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "common_name": detection.get("common_name"),
                    "confidence": detection.get("confidence"),
                }

                # Add to batch
                batch.set(doc_ref, doc_data)

            # Commit the batch
            batch.commit()
            print(
                f"✓ Exported {len(unique_detections)} unique detection(s) to Firebase"
            )

        except Exception as e:
            print(f"Error sending data to Firebase: {e}")
