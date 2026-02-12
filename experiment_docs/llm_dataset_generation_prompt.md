PROVIDER = openai
MODEL = gpt3.5-turbo
TARGET_SENTENCES = 700, 1300, 2000

Generate x number of datasets of **TARGET_SENTENCES restaurant review sentences** (TARGET_REVIEWS reviews) in SemEval XML format for aspect-based sentiment analysis research. The dataset should contain both explicit and implicit aspect mentions.

One dataset per TARGET_SENTENCE size at the bottom of this instruction (comma, separated)

**OUTPUT FORMAT**: Complete XML document with this structure (following SemEval baseline structure):

```xml
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Reviews>
    <Review rid="1001">
        <sentences>
            <sentence id="1001:0">
                <text>The food was absolutely delicious and the service was prompt.</text>
                <Opinions>
                    <Opinion target="food" category="FOOD#QUALITY" polarity="positive" from="4" to="8"/>
                    <Opinion target="service" category="SERVICE#GENERAL" polarity="positive" from="42" to="49"/>
                </Opinions>
            </sentence>
            <sentence id="1001:1">
                <text>I couldn't stop eating, it was so good!</text>
                <Opinions>
                    <Opinion target="NULL" category="FOOD#QUALITY" polarity="positive" from="0" to="0"/>
                </Opinions>
            </sentence>
        </sentences>
    </Review>
    <Review rid="1002">
        <sentences>
            <sentence id="1002:0">
                <text>The atmosphere was cozy but the prices were too high.</text>
                <Opinions>
                    <Opinion target="atmosphere" category="AMBIENCE#GENERAL" polarity="positive" from="4" to="14"/>
                    <Opinion target="prices" category="PRICE#GENERAL" polarity="negative" from="34" to="40"/>
                </Opinions>
            </sentence>
        </sentences>
    </Review>
    <!-- Continue for TARGET_REVIEWS reviews with TARGET_SENTENCES total sentences... -->
</Reviews>
```

**REQUIREMENTS**:
- Generate **TARGET_REVIEWS restaurant reviews** with **TARGET_SENTENCES total sentences** (approximately 5-6 sentences per review)
- Use 5 aspect categories: FOOD#QUALITY, SERVICE#GENERAL, AMBIENCE#GENERAL, PRICE#GENERAL, LOCATION#GENERAL
- Use 3 sentiment polarities with natural variation: positive (roughly 50-70%), negative (roughly 20-40%), neutral (roughly 5-15%) - let the distribution emerge naturally from realistic restaurant experiences
- Include both explicit aspects (with Opinion target="word") and implicit aspects (with Opinion target="NULL")
- Calculate accurate character positions (from/to) for explicit aspects
- Use diverse vocabulary and natural restaurant review language
- Allow for creativity and realism in the reviews; generate lifelike, varied, and believable restaurant experiences
- Each review should be 3-8 sentences long (following SemEval baseline structure)
- Each sentence should be between 8 and 25 words long (to match SemEval dataset style)
- Follow the nested Review->sentences->sentence structure with proper rid and sentence id formatting
- Each sentence MUST BE UNIQUE (no exception!), no two sentences should be the same. Don't just add additional exclamation marks to differentiate sentences. They must be truly unique.

**EXPLICIT ASPECTS**: Direct mentions like "The food was delicious" (Opinion target="food")
**IMPLICIT ASPECTS**: Indirect mentions like "I couldn't stop eating" (Opinion target="NULL")

Generate the complete XML documents now and make them downloadable files instead of printing out.

file naming format: <PROVIDER>-<MODEL>-<TARGET_SENTENCE>.xml