# Restaurant Review Dataset Generator V2

A comprehensive tool for generating high-quality restaurant review datasets in SemEval XML format using various Large Language Model (LLM) providers. Designed specifically for aspect-based sentiment analysis research in the restaurant domain.

**Version 2.0** features a robust three-worker architecture (PROMPTER, XML Validator, PROCESSOR) with batch processing, configurable parameters via `config.json`, enhanced error handling, **payload.json support**, and **parallel execution capabilities**.

## 🚀 Features

### Core Functionality

- **Multi-Provider Support**: OpenAI, Google Gemini, Anthropic, and xAI models
- **Three-Worker Architecture**: PROMPTER → XML Validator → PROCESSOR pipeline
- **Batch Processing**: Generates 5 reviews at a time for better reliability
- **SemEval XML Format**: Generates properly formatted XML with aspect annotations
- **Configurable Parameters**: All settings managed through `config.json`
- **Robust XML Validation**: Filters invalid XML, continues with valid reviews

### Advanced Features

- **Incremental Writing**: Appends each valid batch immediately to file
- **Smart Progress Tracking**: Real-time progress bar with sentence count
- **Error Recovery**: Discards invalid reviews, continues generation
- **Realistic Reviews**: Coherence requirements for multi-sentence reviews
- **Flexible Targeting**: Accepts slight over/under target sentence counts
- **Rich Console Output**: Clear visual feedback with colored status messages

### ✨ New in V2.0

- **Payload.json Support**: Configure complex multi-provider/multi-model setups via JSON
- **Parallel Execution**: Run providers and models concurrently for faster generation
- **Multiple Models per Provider**: Generate datasets from multiple models in one command
- **Enhanced CLI**: Backward-compatible command line with new parallel options

## 📁 Project Structure

```
datasets-generator-v2/
├── .env                    # Environment variables (API keys)
├── config.json             # Configuration parameters
├── payload.json            # Batch configuration (NEW in V2.0)\n├── llm_prompt.md           # LLM prompt template (NEW in V2.1)
├── generate_datasets_v2.py # Main script (V2)
├── output/                 # Generated datasets (default output)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- API keys for your chosen LLM provider

### Dependencies

```bash
# Install from requirements.txt (recommended)
pip install -r requirements.txt

# Or install manually
pip install python-dotenv rich anthropic openai google-generativeai
```

**Note**: Version 2.0 requires Python 3.7+ for asyncio support in parallel execution.

### Configuration Setup

The `config.json` file allows you to customize generation parameters:

```json
{
	"REVIEWS_PER_PROMPT": 5,
	"SENTENCES_PER_REVIEW_FROM": 3,
	"SENTENCES_PER_REVIEW_TO": 8,
	"OPINIONS_PER_SENTENCE_FROM": 1,
	"OPINIONS_PER_SENTENCE_TO": 3,
	"CATEGORIES": ["FOOD#QUALITY", "SERVICE#GENERAL", "AMBIENCE#GENERAL", "PRICE#GENERAL", "LOCATION#GENERAL"],
	"POLARITIES": ["positive", "neutral", "negative"],
	"FILENAME_FORMAT": "{PROVIDER}-{PREFIX}-{TARGET_SIZE}.xml"
}
```

### LLM Prompt Template (New in V2.1)

The LLM prompt is now externalized to `llm_prompt.md` for better transparency and maintainability:\n\n**Template Variables:**\n- `{reviews_per_prompt}`: Number of reviews to generate per batch\n- `{sentences_from}` / `{sentences_to}`: Sentence count range per review\n- `{opinions_from}` / `{opinions_to}`: Opinion count range per sentence\n- `{categories}`: Available aspect categories\n- `{polarities}`: Available sentiment polarities\n\n**Benefits:**\n- **Transparency**: Clear visibility of the exact prompt sent to LLMs\n- **Version Control**: Track prompt changes separately from code\n- **Easy Modification**: Edit prompts without touching Python code\n- **Research Reproducibility**: Share and reference exact prompts used\n\n**Custom Templates:**\nYou can modify `llm_prompt.md` or specify a different template file path in the configuration.\n\n### Environment Setup\nCreate a `.env` file in the `datasets-generator-v2` folder:

```env
OPENAI_API=your_openai_api_key
GEMINI_API=your_gemini_api_key
ANTHROPIC_API=your_anthropic_api_key
```

**Note**: The `.env` file should be in the same directory as `generate_datasets_v2.py` to ensure the script can find your API keys.

## 🏗️ V2 Architecture

### Three-Worker Pipeline

1. **PROMPTER**: Generates batches of 5 reviews using configurable prompts
2. **XML Validator**: Validates XML structure, filters out invalid reviews
3. **PROCESSOR**: Assigns sequential IDs, updates counters, appends to file

### Generation Flow

```
┌─────────────┐    ┌──────────────┐    ┌───────────┐
│  PROMPTER   │───▶│ XML Validator│───▶│ PROCESSOR │
│ (5 reviews) │    │ (filter bad) │    │ (assign   │
│             │    │              │    │  IDs)     │
└─────────────┘    └──────────────┘    └───────────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │ Append to   │
                                       │ XML file    │
                                       └─────────────┘
```

### Supported Models & Pricing

| Provider      | Model                    | Input ($/1M tokens) | Output ($/1M tokens) |
| ------------- | ------------------------ | ------------------- | -------------------- |
| **OpenAI**    | gpt-4o                   | $5.00               | $15.00               |
| **OpenAI**    | gpt-4-turbo              | $10.00              | $30.00               |
| **OpenAI**    | gpt-3.5-turbo            | $1.50               | $2.00                |
| **Anthropic** | claude-sonnet-4-20250514 | $3.00               | $15.00               |
| **Anthropic** | claude-3-sonnet-20240229 | $3.00               | $15.00               |
| **Anthropic** | claude-3-haiku-20240307  | $0.25               | $1.25                |
| **Google**    | gemini-pro               | $0.25               | $0.50                |
| **Google**    | gemini-1.5-pro           | $1.25               | $5.00                |

## 🔧 Usage

### Payload Mode (New in V2.0)

For complex multi-provider setups with parallel execution:

```bash
# Using payload.json
python3 generate_datasets_v2.py --payload payload.json

# Override parallel settings
python3 generate_datasets_v2.py --payload payload.json --parallel-providers
```

### Legacy CLI Mode

#### Basic Usage

```bash
python3 generate_datasets_v2.py --provider anthropic --model claude-sonnet-4-20250514 --sent-sizes 50,100 --prefix restaurant
```

#### Advanced CLI Usage

```bash
python3 generate_datasets_v2.py \
    --provider openai \
    --model gpt-4o \
    --sent-sizes 200,500,1000 \
    --prefix gpt4 \
    --output ./custom-output/
```

#### Multi-Model CLI (Single Provider)

```bash
python3 generate_datasets_v2.py \
    --provider xai \
    --model grok-3,grok-4 \
    --prefix grok3,grok4 \
    --sent-sizes 25,50
```

### Command Line Options

| Option                 | Required | Default       | Description                                   |
| ---------------------- | -------- | ------------- | --------------------------------------------- |
| `--payload`            | ❌       | -             | Path to payload.json for batch configuration  |
| `--provider`           | ✅\*     | -             | LLM provider (openai, google, anthropic, xai) |
| `--model`              | ✅\*     | -             | Specific model name(s), comma-separated       |
| `--sent-sizes`         | ✅\*     | -             | Comma-separated sentence counts               |
| `--prefix`             | ❌       | -             | Prefix(es) for output filenames               |
| `--output`             | ❌       | `output`      | Output directory                              |
| `--config`             | ❌       | `config.json` | Configuration file path                       |
| `--parallel-providers` | ❌       | false         | Run providers in parallel (payload mode only) |
| `--parallel-models`    | ❌       | false         | Run models in parallel (payload mode only)    |

\*Required only when not using `--payload`

### Output Files

**With prefix:**

- `<provider>-<prefix>-<size>.xml`
- Example: `anthropic-restaurant-200.xml`

**Without prefix:**

- `<provider>-<model>-<size>.xml`
- Example: `openai-gpt-4o-300.xml`

## 📊 Generated Dataset Format

The tool generates SemEval-compliant XML with proper aspect annotations:

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
</Reviews>
```

## 🔄 Error Handling

### Robust Recovery System

- **XML Validation**: Invalid reviews are discarded, valid ones are kept
- **Batch Processing**: Failures affect only current batch, not entire generation
- **Incremental Progress**: Each successful batch is immediately written to file
- **Automatic Retry**: Failed batches are retried automatically
- **Graceful Degradation**: Generation continues even with partial batch failures

## 🎯 Quality Assurance

### Review Quality

- **Batch Generation**: 5 reviews per API call for better consistency
- **Coherent Multi-Sentence Reviews**: Related content across sentences
- **Realistic Scenarios**: Avoids mixing unrelated food types
- **Configurable Constraints**: Sentence and opinion counts via config.json

### XML Structure

- **Valid XML**: Strict validation ensures proper structure
- **Sequential IDs**: Reviews (0,1,2...) and sentences (reviewId:0, reviewId:1...)
- **Standard Format**: SemEval-compliant with proper namespaces
- **Incremental Building**: File built progressively, not all at once

## 🏗️ Architecture Notes

### Environment File Location

### Configuration Design

The `config.json` and `.env` files are located in the `datasets-generator-v2` folder for:

- **Self-contained subproject**: All dependencies and configuration in one place
- **Easy customization**: Modify generation parameters without code changes
- **Portability**: Easy to move or distribute independently
- **Clear separation**: Each subproject manages its own settings and API keys

### Alternative Setups

If you prefer to keep `.env` at the project root:

1. Move `.env` to the parent directory
2. Update the script path in `load_dotenv()` if needed
3. Ensure all subprojects can access the shared environment

## 🤝 Contributing

When modifying the script, please:

1. Update this README to reflect changes
2. Update `requirements.txt` if adding new dependencies
3. Test with multiple providers and models
4. Verify cost estimation accuracy
5. Ensure backward compatibility

## 📝 Example Session

```bash
$ python3 generate_datasets_v2.py --provider anthropic --model claude-sonnet-4-20250514 --sent-sizes 25,50 --prefix test

💰 Estimated API Costs
==================================================
Model: claude-sonnet-4-20250514

┌─────────────────┬──────────────────┐
│ Dataset Size    │ Estimated Cost   │
├─────────────────┼──────────────────┤
│ 25 sentences    │         ~$0.0330 │
│ 50 sentences    │         ~$0.0630 │
├─────────────────┼──────────────────┤
│ TOTAL           │         ~$0.0960 │
└─────────────────┴──────────────────┘

⚠️  Important Disclaimer:
┌─────────────────────────────────────────────────────────────┐
│ These are rough estimates based on approximate token usage │
│ Always check your API provider's current pricing!          │
└─────────────────────────────────────────────────────────────┘

Do you want to continue? (Y/n): y

Generating 2 restaurant review datasets
Target sizes: [25, 50]
Provider: anthropic
Model: claude-sonnet-4-20250514
Prefix: test
Output directory: output
Max retries: 3
--------------------------------------------------
Generating restaurant dataset: 25 sentences using claude-sonnet-4-20250514...
Generating 25 sentences ⠋ Saved: output/anthropic-test-25.xml
Generating restaurant dataset: 50 sentences using claude-sonnet-4-20250514...
Generating 50 sentences ⠋ Saved: output/anthropic-test-50.xml
--------------------------------------------------
Generation complete! Successfully created 2/2 datasets.
```

## 🐛 Troubleshooting

### Common Issues

**API Key Errors:**

- Ensure `.env` file is in the `datasets-generator-v2` folder
- Check API key validity and quotas

**Connection Timeouts:**

- Increase `--max-retries` for unreliable connections
- Try smaller dataset sizes first

**Cost Concerns:**

- Use cheaper models like `claude-3-haiku-20240307` for testing
- Start with small dataset sizes to verify costs

### Support

For issues or feature requests, please check the main project repository or create an issue with detailed error messages and steps to reproduce.

## 📅 Version History

### V1.x Series

- **v1.0.0**: Initial release with basic dataset generation
- **v1.1.0**: Added retry logic and cost estimation
- **v1.2.0**: Added sentence count validation and formatted tables
- **v1.3.0**: Organized into dedicated folder structure with README
- **v1.3.1**: Updated pricing table to per-million tokens, added requirements.txt

### V2.x Series

- **v2.0.0**: Complete architectural redesign with three-worker pipeline (PROMPTER, XML Validator, PROCESSOR), batch processing (5 reviews at a time), configurable parameters via `config.json`, robust XML validation, incremental file writing, and enhanced error recovery
- **v2.1.0**: Added payload.json support, parallel execution capabilities (parallel_providers and parallel_models), multiple models per provider support, enhanced CLI with backward compatibility, async/await architecture for improved performance, and externalized LLM prompt template system

---

_Generated datasets are designed for research purposes in aspect-based sentiment analysis. Always verify data quality and comply with your institution's research guidelines._
