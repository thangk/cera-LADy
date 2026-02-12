#!/usr/bin/env python3
"""
Restaurant Review Dataset Generator V2

A comprehensive tool for generating high-quality restaurant review datasets in SemEval XML format 
using various Large Language Model (LLM) providers. This version features a robust three-worker 
architecture with batch processing, XML validation, and configurable parameters.

Version 2.0 - Enhanced multi-threading support, improved error handling, and better performance optimizations.
"""

import argparse
import asyncio
import json
import os
import sys
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from tqdm import tqdm
from dataclasses import dataclass


class ConfigError(Exception):
    """Custom exception for configuration-related errors"""
    pass


class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass


class XMLValidationError(Exception):
    """Custom exception for XML validation errors"""
    pass


@dataclass
class LLMConfig:
    """Configuration for a single LLM provider and its models"""
    provider: str
    models: List[str]
    prefixes: List[str]
    
    def __post_init__(self):
        if len(self.models) != len(self.prefixes):
            raise ConfigError(f"Number of models ({len(self.models)}) must match number of prefixes ({len(self.prefixes)})")


@dataclass
class PayloadConfig:
    """Configuration loaded from payload.json"""
    parallel_providers: bool = False
    parallel_models: bool = False
    config: str = "config.json"
    output_dir: str = "output"
    sent_sizes: List[int] = None
    llms: List[LLMConfig] = None
    
    def __post_init__(self):
        if self.sent_sizes is None:
            self.sent_sizes = []
        if self.llms is None:
            self.llms = []


class DatasetGenerator:
    """Main dataset generator class with three-worker architecture"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the generator with configuration"""
        self.console = Console()
        self._config_path = config_path  # Store for async usage
        self.config = self._load_config(config_path)
        self.reviews_counter = 0
        self.sentences_counter = 0
        self.target_size = 0
        self.output_file = None
        self.llm_client = None
        self.payload_config = None
        self._parallel_mode = False  # Flag to disable progress bars in parallel mode
        
        # Load environment variables
        load_dotenv()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise ConfigError(f"Configuration file not found: {config_path}")
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Validate required config keys
            required_keys = [
                'REVIEWS_PER_PROMPT', 'SENTENCES_PER_REVIEW_FROM', 'SENTENCES_PER_REVIEW_TO',
                'OPINIONS_PER_SENTENCE_FROM', 'OPINIONS_PER_SENTENCE_TO', 
                'CATEGORIES', 'POLARITIES', 'FILENAME_FORMAT'
            ]
            
            for key in required_keys:
                if key not in config:
                    raise ConfigError(f"Missing required configuration key: {key}")
            
            return config
            
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading configuration: {e}")
    
    def _load_payload(self, payload_path: str) -> PayloadConfig:
        """Load payload configuration from JSON file"""
        try:
            payload_file = Path(payload_path)
            if not payload_file.exists():
                raise ConfigError(f"Payload file not found: {payload_path}")
            
            with open(payload_file, 'r') as f:
                payload_data = json.load(f)
            
            # Validate required keys
            if 'llms' not in payload_data:
                raise ConfigError("Missing required key 'llms' in payload file")
            
            # Parse LLM configurations
            llm_configs = []
            for llm_data in payload_data['llms']:
                if not all(key in llm_data for key in ['provider', 'models', 'prefixes']):
                    raise ConfigError("Each LLM config must have 'provider', 'models', and 'prefixes'")
                
                llm_config = LLMConfig(
                    provider=llm_data['provider'],
                    models=llm_data['models'],
                    prefixes=llm_data['prefixes']
                )
                llm_configs.append(llm_config)
            
            # Create payload config
            payload_config = PayloadConfig(
                parallel_providers=payload_data.get('parallel_providers', False),
                parallel_models=payload_data.get('parallel_models', False),
                config=payload_data.get('config', 'config.json'),
                output_dir=payload_data.get('output_dir', 'output'),
                sent_sizes=payload_data.get('sent_sizes', []),
                llms=llm_configs
            )
            
            return payload_config
            
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in payload file: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading payload: {e}")
    
    def _load_prompt_template(self, template_path: str = "llm_prompt.md") -> str:
        """Load prompt template from markdown file"""
        try:
            template_file = Path(template_path)
            
            # If relative path, look in same directory as script
            if not template_file.is_absolute():
                # Use current working directory if __file__ not available (in async context)
                try:
                    script_dir = Path(__file__).parent
                except NameError:
                    script_dir = Path.cwd()
                template_file = script_dir / template_path
            
            if not template_file.exists():
                raise ConfigError(f"Prompt template file not found: {template_file}")
            
            with open(template_file, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            return template_content
            
        except Exception as e:
            raise ConfigError(f"Error loading prompt template: {e}")
    
    def _setup_llm_client(self, provider: str, model: str):
        """Setup LLM client based on provider"""
        if provider == "openai":
            import openai
            api_key = os.getenv("OPENAI_API")
            if not api_key:
                raise LLMError("OPENAI_API key not found in environment variables")
            self.llm_client = openai.OpenAI(api_key=api_key)
            
        elif provider == "anthropic":
            import anthropic
            api_key = os.getenv("ANTHROPIC_API")
            if not api_key:
                raise LLMError("ANTHROPIC_API key not found in environment variables")
            self.llm_client = anthropic.Anthropic(api_key=api_key)
            
        elif provider == "google":
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API")
            if not api_key:
                raise LLMError("GEMINI_API key not found in environment variables")
            genai.configure(api_key=api_key)
            self.llm_client = genai.GenerativeModel(model)
            
        elif provider == "xai":
            import openai
            api_key = os.getenv("XAI_API")
            if not api_key:
                raise LLMError("XAI_API key not found in environment variables")
            self.llm_client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
        else:
            raise LLMError(f"Unsupported provider: {provider}")
    
    async def _setup_llm_client_async(self, provider: str, model: str):
        """Async version of LLM client setup"""
        # For now, just call the sync version in an executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._setup_llm_client, provider, model)
    
    def _create_prompt(self) -> str:
        """Create the prompt for generating reviews based on configuration"""
        # Load template from file
        template = self._load_prompt_template()
        
        # Prepare template variables from config
        reviews_per_prompt = self.config['REVIEWS_PER_PROMPT']
        sentences_from = self.config['SENTENCES_PER_REVIEW_FROM']
        sentences_to = self.config['SENTENCES_PER_REVIEW_TO']
        opinions_from = self.config['OPINIONS_PER_SENTENCE_FROM']
        opinions_to = self.config['OPINIONS_PER_SENTENCE_TO']
        categories = ', '.join(self.config['CATEGORIES'])
        polarities = ', '.join(self.config['POLARITIES'])
        
        # Substitute variables in template
        prompt = template.format(
            reviews_per_prompt=reviews_per_prompt,
            sentences_from=sentences_from,
            sentences_to=sentences_to,
            opinions_from=opinions_from,
            opinions_to=opinions_to,
            categories=categories,
            polarities=polarities
        )
        
        return prompt
    
    def _prompter(self, provider: str, model: str) -> str:
        """PROMPTER worker: Generate batch of reviews from LLM"""
        prompt = self._create_prompt()
        
        try:
            if provider == "openai" or provider == "xai":
                self.console.print(f"[dim]Making request to {provider} {model}...[/dim]")
                response = self.llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=4000
                )
                content = response.choices[0].message.content.strip()
                self.console.print(f"[dim]✓ Received response ({len(content)} chars)[/dim]")
                
            elif provider == "anthropic":
                self.console.print(f"[dim]Making request to {provider} {model}...[/dim]")
                response = self.llm_client.messages.create(
                    model=model,
                    max_tokens=4000,
                    temperature=0.8,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text.strip()
                self.console.print(f"[dim]✓ Received response ({len(content)} chars)[/dim]")
                
            elif provider == "google":
                self.console.print(f"[dim]Making request to {provider} {model}...[/dim]")
                response = self.llm_client.generate_content(
                    prompt,
                    generation_config={'temperature': 0.8, 'max_output_tokens': 4000}
                )
                
                # Check if response was blocked or had issues
                if hasattr(response, 'prompt_feedback'):
                    feedback = response.prompt_feedback
                    if hasattr(feedback, 'block_reason') and feedback.block_reason:
                        self.console.print(f"[red]⚠️  Prompt blocked: {feedback.block_reason}[/red]")
                        raise LLMError(f"Prompt blocked by safety filters: {feedback.block_reason}")
                
                if not hasattr(response, 'text') or not response.text:
                    # Check candidates for finish reason
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'finish_reason'):
                            self.console.print(f"[red]⚠️  Generation stopped: {candidate.finish_reason}[/red]")
                            raise LLMError(f"Generation stopped: {candidate.finish_reason}")
                    raise LLMError("No text content received from Gemini")
                
                content = response.text.strip()
                self.console.print(f"[dim]✓ Received response ({len(content)} chars)[/dim]")
            
            # Remove code block markers if present
            content = self._clean_llm_output(content)
            
            return content
                
        except Exception as e:
            self.console.print(f"[red]❌ LLM Error ({provider}): {str(e)}[/red]")
            # Log more details about the error
            if hasattr(e, '__class__'):
                self.console.print(f"[red]Error type: {e.__class__.__name__}[/red]")
            raise LLMError(f"Error generating content from {provider}: {str(e)}")
    
    def _clean_llm_output(self, content: str) -> str:
        """Remove code block markers and other unwanted formatting from LLM output"""
        # Remove code block markers (``` at start and end)
        content = re.sub(r'^```[\w]*\n?', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n?```$', '', content, flags=re.MULTILINE)
        
        # Remove any remaining ``` that might be in the middle
        content = content.replace('```', '')
        
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        return content
    
    def _xml_validator(self, xml_content: str) -> List[ET.Element]:
        """XML Validator worker: Validate and parse XML, return valid reviews"""
        valid_reviews = []
        
        # Wrap content in a root element for parsing
        wrapped_content = f"<root>{xml_content}</root>"
        
        try:
            root = ET.fromstring(wrapped_content)
            reviews = root.findall('Review')
            
            for review in reviews:
                try:
                    # Validate review structure
                    if not review.get('rid'):
                        continue
                    
                    sentences_elem = review.find('sentences')
                    if sentences_elem is None:
                        continue
                    
                    sentences = sentences_elem.findall('sentence')
                    if not sentences:
                        continue
                    
                    # Validate each sentence
                    valid_sentences = []
                    for sentence in sentences:
                        if not sentence.get('id'):
                            continue
                        
                        text_elem = sentence.find('text')
                        if text_elem is None or not text_elem.text:
                            continue
                        
                        opinions_elem = sentence.find('Opinions')
                        if opinions_elem is None:
                            continue
                        
                        opinions = opinions_elem.findall('Opinion')
                        if not opinions:
                            continue
                        
                        # Validate each opinion
                        valid_opinions = []
                        for opinion in opinions:
                            required_attrs = ['target', 'category', 'polarity', 'from', 'to']
                            if all(opinion.get(attr) for attr in required_attrs):
                                if opinion.get('category') in self.config['CATEGORIES']:
                                    if opinion.get('polarity') in self.config['POLARITIES']:
                                        valid_opinions.append(opinion)
                        
                        if valid_opinions:
                            # Update opinions in sentence
                            opinions_elem.clear()
                            for valid_opinion in valid_opinions:
                                opinions_elem.append(valid_opinion)
                            valid_sentences.append(sentence)
                    
                    if valid_sentences:
                        # Update sentences in review
                        sentences_elem.clear()
                        for valid_sentence in valid_sentences:
                            sentences_elem.append(valid_sentence)
                        valid_reviews.append(review)
                        
                except ET.ParseError:
                    continue
                    
        except ET.ParseError as e:
            self.console.print(f"[yellow]Warning: XML parsing error: {e}[/yellow]")
            
        return valid_reviews
    
    def _processor(self, valid_reviews: List[ET.Element]) -> int:
        """PROCESSOR worker: Assign IDs, update counters, append to file"""
        sentences_processed = 0
        
        for review in valid_reviews:
            # Assign review ID
            review.set('rid', str(self.reviews_counter))
            
            sentences_elem = review.find('sentences')
            sentences = sentences_elem.findall('sentence')
            
            # Assign sentence IDs
            for sentence_idx, sentence in enumerate(sentences):
                sentence_id = f"{self.reviews_counter}:{sentence_idx}"
                sentence.set('id', sentence_id)
                sentences_processed += 1
                self.sentences_counter += 1
            
            # Append review to file
            review_xml = ET.tostring(review, encoding='unicode')
            # Format the XML nicely
            formatted_xml = self._format_xml(review_xml)
            
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(formatted_xml + '\n')
            
            self.reviews_counter += 1
        
        return sentences_processed
    
    def _format_xml(self, xml_string: str) -> str:
        """Format XML string with proper indentation"""
        try:
            # Parse and reformat for pretty printing
            element = ET.fromstring(xml_string)
            self._indent(element)
            return ET.tostring(element, encoding='unicode')
        except:
            return xml_string
    
    def _indent(self, elem, level=0):
        """Add indentation to XML elements"""
        i = "\n" + level * "    "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "    "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    def _initialize_output_file(self, filename: str):
        """Initialize output file with XML header and opening tag"""
        self.output_file = filename
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n')
            f.write('<Reviews>\n')
    
    def _finalize_output_file(self):
        """Add closing tag to output file"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write('</Reviews>\n')
    
    def generate_dataset(self, provider: str, model: str, target_size: int, 
                        prefix: str = None, output_dir: str = "output") -> str:
        """Main method to generate a complete dataset"""
        
        # Setup
        self.target_size = target_size
        self.reviews_counter = 0
        self.sentences_counter = 0
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate filename
        if prefix:
            filename = self.config['FILENAME_FORMAT'].format(
                PROVIDER=provider, PREFIX=prefix, TARGET_SIZE=target_size
            )
        else:
            filename = self.config['FILENAME_FORMAT'].format(
                PROVIDER=provider, PREFIX=model.replace('.', '-'), TARGET_SIZE=target_size
            )
        
        output_path = Path(output_dir) / filename
        
        # Initialize output file
        self._initialize_output_file(str(output_path))
        
        # Setup LLM client
        self._setup_llm_client(provider, model)
        
        current_batch = 0
        
        # Create beautiful tqdm progress bar
        provider_colors = {
            'anthropic': '\033[95m',  # Magenta
            'google': '\033[92m',     # Green  
            'openai': '\033[94m',     # Blue
            'xai': '\033[93m'         # Yellow
        }
        reset_color = '\033[0m'
        
        provider_color = provider_colors.get(provider, '\033[96m')  # Default cyan
        model_short = model.split('-')[-1] if '-' in model else model
        
        # Create colored description
        colored_desc = f"{provider_color}{provider:<10}{reset_color} {model_short:<15}"
        
        pbar = tqdm(
            total=target_size,
            desc=colored_desc,
            unit=" sents",
            leave=True,
            position=getattr(self, '_tqdm_position', 0),
            ncols=130,
            bar_format='{desc} │ {percentage:3.0f}% │{bar:35}│ {n_fmt}/{total_fmt}{unit} │ {rate_fmt} │ {elapsed}<{remaining}',
            ascii=False,
            colour='green'
        )
        
        try:
            while True:
                # Stop when we have enough sentences
                if self.sentences_counter >= target_size:
                    break
                
                current_batch += 1
                
                try:
                    # Update progress description for current batch
                    batch_status = f"Batch {current_batch}"
                    pbar.set_description(f"{colored_desc} {batch_status:<12}")
                    
                    # PROMPTER: Generate reviews
                    xml_content = self._prompter(provider, model)
                    
                    # XML Validator: Validate and filter
                    valid_reviews = self._xml_validator(xml_content)
                    
                    if not valid_reviews:
                        retry_status = f"Batch {current_batch} Retry"
                        pbar.set_description(f"{colored_desc} {retry_status:<12}")
                        current_batch -= 1  # Don't count failed attempts
                        continue
                    
                    # PROCESSOR: Assign IDs and write to file
                    sentences_processed = self._processor(valid_reviews)
                    
                    # Update progress
                    pbar.update(sentences_processed)
                    done_status = f"✓ Batch {current_batch}"
                    pbar.set_description(f"{colored_desc} {done_status:<12}")
                    
                except Exception as e:
                    self.console.print(f"[red]❌ Batch {current_batch} failed: {str(e)}[/red]")
                    error_status = f"❌ Error B{current_batch}"
                    pbar.set_description(f"{colored_desc} {error_status:<12}")
                    
                    # Log detailed error information
                    if isinstance(e, LLMError):
                        self.console.print(f"[yellow]LLM Error details: {str(e)}[/yellow]")
                    else:
                        self.console.print(f"[yellow]Unexpected error: {e.__class__.__name__}: {str(e)}[/yellow]")
                    
                    current_batch -= 1  # Don't count failed attempts
                    
                    # Add a short delay before retrying
                    time.sleep(2)
                    continue
        
        finally:
            # Close the progress bar
            pbar.close()
        
        # Finalize file
        self._finalize_output_file()
        
        # Return output path without individual completion messages to avoid UI breaking
        return str(output_path)
    
    async def generate_dataset_async(self, provider: str, model: str, target_size: int, 
                                   prefix: str = None, output_dir: str = "output", 
                                   tqdm_position: int = 0) -> str:
        """Async version of dataset generation"""
        # Create a new generator instance for each async task to avoid client conflicts
        generator = DatasetGenerator(self._config_path)
        # Set tqdm position for multiple progress bars
        generator._tqdm_position = tqdm_position
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            generator.generate_dataset, 
            provider, model, target_size, prefix, output_dir
        )
    
    async def generate_models_for_provider(self, llm_config: LLMConfig, target_sizes: List[int], 
                                         output_dir: str, parallel_models: bool = False) -> List[str]:
        """Generate datasets for all models of a single provider"""
        results = []
        
        if parallel_models:
            # Run all models for this provider in parallel
            tasks = []
            position = 0
            for model, prefix in zip(llm_config.models, llm_config.prefixes):
                for target_size in target_sizes:
                    task = self.generate_dataset_async(
                        provider=llm_config.provider,
                        model=model,
                        target_size=target_size,
                        prefix=prefix,
                        output_dir=output_dir,
                        tqdm_position=position
                    )
                    tasks.append(task)
                    position += 1
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Run models sequentially
            for model, prefix in zip(llm_config.models, llm_config.prefixes):
                for target_size in target_sizes:
                    try:
                        result = await self.generate_dataset_async(
                            provider=llm_config.provider,
                            model=model,
                            target_size=target_size,
                            prefix=prefix,
                            output_dir=output_dir,
                            tqdm_position=0  # Sequential mode can use position 0
                        )
                        results.append(result)
                    except Exception as e:
                        self.console.print(f"[red]Error generating {model}: {e}[/red]")
                        results.append(f"Error: {e}")
        
        return results
    
    async def generate_from_payload(self, payload_config: PayloadConfig) -> Dict[str, List[str]]:
        """Generate datasets from payload configuration"""
        # Update generator config if specified in payload
        if payload_config.config != "config.json":
            self.config = self._load_config(payload_config.config)
        
        results = {}
        
        if payload_config.parallel_providers:
            # Run all providers in parallel with multiple tqdm progress bars
            self.console.print("[green]Running providers in parallel with individual progress bars[/green]")
            
            # Calculate tqdm positions for each provider/model combination
            tasks = []
            position = 0
            
            for llm_config in payload_config.llms:
                self.console.print(f"[dim]Starting {llm_config.provider} ({', '.join(llm_config.models)})...[/dim]")
                
                if payload_config.parallel_models:
                    # Each model gets its own position
                    provider_tasks = []
                    for model, prefix in zip(llm_config.models, llm_config.prefixes):
                        for target_size in payload_config.sent_sizes:
                            task = self.generate_dataset_async(
                                provider=llm_config.provider,
                                model=model,
                                target_size=target_size,
                                prefix=prefix,
                                output_dir=payload_config.output_dir,
                                tqdm_position=position
                            )
                            provider_tasks.append(task)
                            position += 1
                    
                    # Gather all tasks for this provider
                    provider_task = asyncio.gather(*provider_tasks, return_exceptions=True)
                    tasks.append((llm_config.provider, provider_task))
                else:
                    # Sequential models for this provider, but provider runs in parallel
                    task = self.generate_models_for_provider(
                        llm_config=llm_config,
                        target_sizes=payload_config.sent_sizes,
                        output_dir=payload_config.output_dir,
                        parallel_models=False
                    )
                    tasks.append((llm_config.provider, task))
                    position += len(llm_config.models) * len(payload_config.sent_sizes)
            
            # Wait for all providers to complete
            completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # Map results back to providers
            for (provider, _), result in zip(tasks, completed_tasks):
                results[provider] = result if not isinstance(result, Exception) else [f"Error: {result}"]
        else:
            # Run providers sequentially - progress bars work fine
            for llm_config in payload_config.llms:
                self.console.print(f"\n[bold cyan]Starting provider: {llm_config.provider}[/bold cyan]")
                try:
                    provider_results = await self.generate_models_for_provider(
                        llm_config=llm_config,
                        target_sizes=payload_config.sent_sizes,
                        output_dir=payload_config.output_dir,
                        parallel_models=payload_config.parallel_models
                    )
                    results[llm_config.provider] = provider_results
                    self.console.print(f"[bold green]✅ {llm_config.provider} completed[/bold green]")
                except Exception as e:
                    self.console.print(f"[red]Error with provider {llm_config.provider}: {e}[/red]")
                    results[llm_config.provider] = [f"Error: {e}"]
        
        return results


def main():
    """Main function to handle command line interface"""
    parser = argparse.ArgumentParser(
        description="Restaurant Review Dataset Generator V2\n\n" +
                   "Multiple Models Support:\n" +
                   "  Same provider, multiple models:\n" +
                   "    --provider openai --model gpt-3.5-turbo,gpt-4o --prefix gpt3.5turbo,gpt4o\n" +
                   "  Different providers:\n" +
                   "    --provider openai,xai --model gpt-4o,grok4 --prefix gpt4o,grok4\n" +
                   "  Single prefix for all models:\n" +
                   "    --provider openai --model gpt-3.5-turbo,gpt-4o --prefix mytest\n" +
                   "  Payload file support:\n" +
                   "    --payload path/to/payload.json\n" +
                   "  Each model generates all sentence sizes before moving to next model.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add payload argument as an alternative to individual arguments
    parser.add_argument("--payload", 
                       help="Path to payload.json file for batch configuration")
    
    # Legacy CLI arguments (maintained for backward compatibility)
    parser.add_argument("--provider", 
                       help="LLM provider(s) - single provider or comma-separated list (e.g., openai,xai). Valid providers: openai, anthropic, google, xai")
    parser.add_argument("--model", 
                       help="Model name(s) - single model or comma-separated list (e.g., gpt-4o,grok4)")
    parser.add_argument("--sent-sizes",
                       help="Comma-separated sentence counts (e.g., 50,100,200)")
    parser.add_argument("--prefix", 
                       help="Prefix(es) for output filenames - single prefix (used for all models) or comma-separated list matching models")
    parser.add_argument("--output", default="output",
                       help="Output directory (default: output)")
    parser.add_argument("--config", default="config.json",
                       help="Configuration file path (default: config.json)")
    
    # Parallel execution flags
    parser.add_argument("--parallel-providers", action="store_true",
                       help="Run different providers in parallel (only with --payload)")
    parser.add_argument("--parallel-models", action="store_true",
                       help="Run different models within same provider in parallel (only with --payload)")
    
    args = parser.parse_args()
    
    console = Console()
    
    # Handle payload mode vs legacy CLI mode
    if args.payload:
        # Payload mode
        try:
            generator = DatasetGenerator(args.config)
            payload_config = generator._load_payload(args.payload)
            
            # Override parallel settings if specified via CLI
            if args.parallel_providers:
                payload_config.parallel_providers = True
            if args.parallel_models:
                payload_config.parallel_models = True
            
            console.print()
            console.print(Panel.fit(
                f"[bold blue]Restaurant Review Dataset Generator V2 - Payload Mode[/bold blue]\n"
                f"Payload file: {args.payload}\n"
                f"Parallel providers: {payload_config.parallel_providers}\n"
                f"Parallel models: {payload_config.parallel_models}\n"
                f"Target sizes: {payload_config.sent_sizes}\n"
                f"Output directory: {payload_config.output_dir}\n"
                f"Total providers: {len(payload_config.llms)}",
                title="Payload Configuration"
            ))
            
            # Run payload generation
            results = asyncio.run(generator.generate_from_payload(payload_config))
            
            # Display final results in a comprehensive summary table
            console.print(f"\n[bold green]🚀 All datasets generated successfully![/bold green]")
            
            # Create summary table
            table = Table(title="Dataset Generation Summary", show_header=True, header_style="bold cyan")
            table.add_column("Provider", style="cyan", min_width=12)
            table.add_column("Dataset File", style="green", min_width=25)
            table.add_column("Status", style="bright_green", min_width=10)
            
            total_datasets = 0
            successful_datasets = 0
            
            for provider, provider_results in results.items():
                for result in provider_results:
                    total_datasets += 1
                    if isinstance(result, str) and not result.startswith("Error:"):
                        # Extract just the filename from the full path
                        filename = result.split('/')[-1] if '/' in result else result
                        table.add_row(provider, filename, "✅ Success")
                        successful_datasets += 1
                    else:
                        error_msg = result if isinstance(result, str) else str(result)
                        table.add_row(provider, "N/A", f"❌ {error_msg[:30]}..." if len(error_msg) > 30 else f"❌ {error_msg}")
            
            console.print(table)
            console.print(f"\n[bold]Summary: {successful_datasets}/{total_datasets} datasets generated successfully[/bold]")
            
        except (ConfigError, LLMError) as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[bold red]Unexpected error: {e}[/bold red]")
            sys.exit(1)
        
        return
    
    # Legacy CLI mode - require all arguments
    if not all([args.provider, args.model, args.sent_sizes]):
        console.print("[bold red]Error: --provider, --model, and --sent-sizes are required when not using --payload[/bold red]")
        sys.exit(1)
    
    # Parse target sizes
    try:
        target_sizes = [int(size.strip()) for size in args.sent_sizes.split(',')]
    except ValueError:
        console.print("[red]Error: Invalid sentence sizes. Use comma-separated integers.[/red]")
        sys.exit(1)
    
    # Parse providers
    providers = [provider.strip() for provider in args.provider.split(',')]
    valid_providers = ["openai", "anthropic", "google", "xai"]
    for provider in providers:
        if provider not in valid_providers:
            console.print(f"[red]Error: Invalid provider '{provider}'. Valid providers: {', '.join(valid_providers)}[/red]")
            sys.exit(1)
    
    # Parse models
    models = [model.strip() for model in args.model.split(',')]
    
    # Validate provider-model pairs
    if len(providers) == 1 and len(models) > 1:
        # Use single provider for all models
        providers = providers * len(models)
    elif len(providers) != len(models):
        console.print(f"[red]Error: Number of providers ({len(providers)}) must be 1 or match number of models ({len(models)})[/red]")
        sys.exit(1)
    
    # Parse prefixes - handle single prefix for multiple models
    if args.prefix:
        prefixes = [prefix.strip() for prefix in args.prefix.split(',')]
        if len(prefixes) == 1 and len(models) > 1:
            # Use the single prefix for all models
            prefixes = prefixes * len(models)
        elif len(prefixes) != len(models):
            console.print(f"[red]Error: Number of prefixes ({len(prefixes)}) must be 1 or match number of models ({len(models)})[/red]")
            sys.exit(1)
    else:
        prefixes = [None] * len(models)
    
    try:
        # Initialize generator
        generator = DatasetGenerator(args.config)
        
        console.print()
        console.print(Panel.fit(
            f"[bold blue]Restaurant Review Dataset Generator V2[/bold blue]\n"
            f"Providers: {providers}\n"
            f"Models: {models}\n"
            f"Prefixes: {prefixes}\n"
            f"Target sizes: {target_sizes}\n"
            f"Output directory: {args.output}",
            title="Generation Setup"
        ))
        
        # Generate datasets for each provider-model pair
        total_datasets = len(models) * len(target_sizes)
        dataset_counter = 0
        generated_files = []
        
        for model_idx, (provider, model, prefix) in enumerate(zip(providers, models, prefixes)):
            console.print(f"\n[bold magenta]{'='*80}[/bold magenta]")
            console.print(f"[bold magenta]Starting Model {model_idx+1}/{len(models)}: {provider} - {model}[/bold magenta]")
            console.print(f"[bold magenta]{'='*80}[/bold magenta]")
            
            # Generate datasets for current model
            for size_idx, target_size in enumerate(target_sizes):
                dataset_counter += 1
                console.print(f"\n[bold]{'='*60}[/bold]")
                console.print(f"[bold cyan]Dataset {dataset_counter}/{total_datasets}: {provider} {model} - {target_size} sentences[/bold cyan]")
                console.print(f"[bold]{'='*60}[/bold]")
                
                try:
                    output_file = generator.generate_dataset(
                        provider=provider,
                        model=model,
                        target_size=target_size,
                        prefix=prefix,
                        output_dir=args.output
                    )
                    generated_files.append((provider, model, target_size, output_file, "Success"))
                except Exception as e:
                    console.print(f"[red]Error generating dataset: {e}[/red]")
                    generated_files.append((provider, model, target_size, "N/A", f"Error: {str(e)[:30]}..."))
        
        # Display final summary table
        console.print(f"\n[bold green]🚀 All datasets generation completed![/bold green]")
        
        # Create summary table
        table = Table(title="Dataset Generation Summary", show_header=True, header_style="bold cyan")
        table.add_column("Provider", style="cyan", min_width=12)
        table.add_column("Model", style="blue", min_width=20)
        table.add_column("Sentences", style="yellow", min_width=10)
        table.add_column("Output File", style="green", min_width=25)
        table.add_column("Status", style="bright_green", min_width=10)
        
        successful_datasets = 0
        
        for provider, model, target_size, output_file, status in generated_files:
            # Extract just the filename from the full path
            filename = output_file.split('/')[-1] if '/' in output_file and output_file != "N/A" else output_file
            if status == "Success":
                table.add_row(provider, model, str(target_size), filename, "✅ Success")
                successful_datasets += 1
            else:
                table.add_row(provider, model, str(target_size), "N/A", f"❌ {status}")
        
        console.print(table)
        console.print(f"\n[bold]Summary: {successful_datasets}/{total_datasets} datasets generated successfully across {len(models)} models![/bold]")
        
    except (ConfigError, LLMError) as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Generation cancelled by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()