# Imports the Google Cloud Translation library
from google.cloud import translate


# Initialize Translation client
def translate_text(
    text: str = "YOUR_TEXT_TO_TRANSLATE", project_id: str = "apt-market-430913-t8"
) -> translate.TranslationServiceClient:
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "en-US",
            "target_language_code": "id",
        }
    )

    # Display the translation for each input text provided
    for translation in response.translations:
        return translation.translated_text


print(translate_text("im cooking"))
