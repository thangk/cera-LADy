# Restaurant Review Generation Prompt

Generate me {reviews_per_prompt} unique restaurant reviews that sound as realistic as possible in the following XML tags.

## Example Structure

A single review looks like this (generate {reviews_per_prompt} of these):

```xml
<Review rid="REVIEW_ID">
    <sentences>
        <sentence id="REVIEW_ID:SENTENCE_ID">
            <text>[sentence 1 of this review]</text>
            <Opinions>
                <Opinion target="NULL" category="[CATEGORY]" polarity="[POLARITY]" from="0" to="0" />
            </Opinions>
        </sentence>
    </sentences>
</Review>
```

## Requirements

- each review can have between {sentences_from} and {sentences_to} sentences.
- each sentence can have between {opinions_from} and {opinions_to} opinions.
- if a review has more than 1 sentence, make sure they're coherent (avoid talk of unrelated food products like noodles and fastfood as they may not be sold at the same place, thus unrealistic to talk about them between several sentences)
- pick a category from these 5 possible options for the sentence: {categories}
- pick an appropriate polarity from these 3 possible options for the sentence: {polarities}
- target is always "NULL"
- from and to are both always "0"

## IMPORTANT OUTPUT FORMAT

- Generate ONLY the XML content - no additional text, explanations, or formatting
- Do NOT include any text before the first <Review> tag
- Do NOT include any text between <Review> tags (no numbers, dashes, separators, etc.)
- Do NOT include any text after the last </Review> tag
- Each <Review> should be immediately followed by the next <Review> with no content in between

Your output should start directly with the first <Review> tag and end with the last </Review> tag.