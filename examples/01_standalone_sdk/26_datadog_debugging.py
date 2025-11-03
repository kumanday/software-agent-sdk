#!/usr/bin/env python3
"""
Datadog Debugging Example

This example demonstrates how to use the OpenHands agent to debug errors
logged in Datadog.
The agent will:
1. Query Datadog logs to understand the error using curl commands
2. Clone relevant GitHub repositories using git commands
3. Analyze the codebase to identify potential causes
4. Attempt to reproduce the error
5. Optionally create a draft PR with a fix

Usage:
    python 26_datadog_debugging.py --query "status:error service:deploy" \\
        --repos "All-Hands-AI/OpenHands,All-Hands-AI/deploy"

Environment Variables Required:
    - DD_API_KEY: Your Datadog API key
    - DD_APP_KEY: Your Datadog application key
    - DD_SITE: (optional) Datadog site (e.g., datadoghq.com, datadoghq.eu)
    - GITHUB_TOKEN: Your GitHub personal access token
    - LLM_API_KEY: API key for the LLM service
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    LLMConvertibleEvent,
    Message,
    TextContent,
    get_logger,
)
from openhands.sdk.tool import Tool, register_tool
from openhands.tools.execute_bash import BashTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool


logger = get_logger(__name__)


def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        "DD_API_KEY",
        "DD_APP_KEY",
        "GITHUB_TOKEN",
        "LLM_API_KEY",
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set the following environment variables:")
        for var in missing_vars:
            print(f"  export {var}=your_key_here")
        return False

    return True


def fetch_datadog_errors(query: str, working_dir: Path, limit: int = 5) -> Path:
    """
    Fetch error examples from Datadog Error Tracking and save to a JSON file.

    Args:
        query: Datadog query string (can be error tracking query or issue ID)
        working_dir: Directory to save the error examples
        limit: Maximum number of error examples to fetch (default: 5)

    Returns:
        Path to the JSON file containing error examples
    """
    dd_api_key = os.getenv("DD_API_KEY")
    dd_app_key = os.getenv("DD_APP_KEY")
    dd_site = os.getenv("DD_SITE", "datadoghq.com")

    # Use Error Tracking Search API
    api_url = f"https://api.{dd_site}/api/v2/error-tracking/issues/search"

    # Calculate timestamps (30 days back)
    now = int(datetime.now().timestamp() * 1000)
    thirty_days_ago = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)

    # Build the request body for Error Tracking API
    request_body = {
        "data": {
            "attributes": {
                "query": query,
                "from": thirty_days_ago,
                "to": now,
                "track": "logs",  # Track errors from logs
            },
            "type": "search_request",
        }
    }

    print(f"📡 Fetching up to {limit} error tracking issues from Datadog...")
    print(f"   Query: {query}")
    print(f"   API: {api_url}")

    # Add include parameter to get full issue details
    api_url_with_include = f"{api_url}?include=issue"

    # Use curl to fetch data
    curl_cmd = [
        "curl",
        "-X",
        "POST",
        api_url_with_include,
        "-H",
        "Content-Type: application/json",
        "-H",
        f"DD-API-KEY: {dd_api_key}",
        "-H",
        f"DD-APPLICATION-KEY: {dd_app_key}",
        "-d",
        json.dumps(request_body),
        "-s",  # Silent mode
    ]

    try:
        result = subprocess.run(
            curl_cmd, capture_output=True, text=True, check=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        print("❌ Error: Request to Datadog API timed out")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error fetching from Datadog API: {e}")
        print(f"   stderr: {e.stderr}")
        sys.exit(1)

    try:
        response_data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing Datadog API response: {e}")
        print(f"   Response: {result.stdout[:500]}")
        sys.exit(1)

    # Check for API errors
    if "errors" in response_data:
        print(f"❌ Datadog API error: {response_data['errors']}")
        sys.exit(1)

    # Extract and format error tracking issues
    error_examples = []
    search_results = response_data.get("data", [])
    included_details = {item["id"]: item for item in response_data.get("included", [])}

    if search_results:
        for idx, search_result in enumerate(search_results[:limit], 1):
            issue_id = search_result.get("id")
            search_attrs = search_result.get("attributes", {})

            # Get detailed issue info from included section
            issue_details = included_details.get(issue_id, {})
            issue_attrs = issue_details.get("attributes", {})

            error_example = {
                "example_number": idx,
                "issue_id": issue_id,
                "total_count": search_attrs.get("total_count"),
                "impacted_users": search_attrs.get("impacted_users"),
                "impacted_sessions": search_attrs.get("impacted_sessions"),
                "service": issue_attrs.get("service"),
                "error_type": issue_attrs.get("error_type"),
                "error_message": issue_attrs.get("error_message", ""),
                "file_path": issue_attrs.get("file_path"),
                "function_name": issue_attrs.get("function_name"),
                "first_seen": issue_attrs.get("first_seen"),
                "last_seen": issue_attrs.get("last_seen"),
                "state": issue_attrs.get("state"),
                "platform": issue_attrs.get("platform"),
                "languages": issue_attrs.get("languages", []),
            }
            error_examples.append(error_example)

    # Save to file
    errors_file = working_dir / "datadog_errors.json"
    with open(errors_file, "w") as f:
        json.dump(
            {
                "query": query,
                "fetch_time": "now",
                "total_examples": len(error_examples),
                "examples": error_examples,
            },
            f,
            indent=2,
        )

    print(f"✅ Fetched {len(error_examples)} error examples")
    print(f"📄 Saved to: {errors_file}")
    return errors_file


def create_debugging_prompt(query: str, repos: list[str], errors_file: Path) -> str:
    """Create the debugging prompt for the agent."""
    repos_list = "\n".join(f"- {repo}" for repo in repos)
    dd_site = os.getenv("DD_SITE", "datadoghq.com")
    error_tracking_url = f"https://api.{dd_site}/api/v2/error-tracking/issues/search"
    logs_url = f"https://api.{dd_site}/api/v2/logs/events/search"

    prompt = (
        "Your task is to debug an error from Datadog Error Tracking to find "
        "out why it is happening.\n\n"
        "## Error Tracking Issues\n\n"
        f"I have already fetched error tracking issues and saved them to: "
        f"`{errors_file}`\n\n"
        "This JSON file contains:\n"
        "- `query`: The Datadog query used to fetch these errors\n"
        "- `total_examples`: Number of error tracking issues in the file\n"
        "- `examples`: Array of error tracking issues, where each has:\n"
        "  - `issue_id`: Unique identifier for the aggregated error issue\n"
        "  - `total_count`: Total number of error occurrences\n"
        "  - `impacted_users`: Number of users affected\n"
        "  - `service`: Service name where errors occurred\n"
        "  - `error_type`: Type of error (e.g., exception class)\n"
        "  - `error_message`: Error message text\n"
        "  - `file_path`: Source file where error occurred\n"
        "  - `function_name`: Function where error occurred\n"
        "  - `first_seen`: Timestamp when first seen (milliseconds)\n"
        "  - `last_seen`: Timestamp when last seen (milliseconds)\n"
        "  - `state`: Issue state (OPEN, ACKNOWLEDGED, RESOLVED, etc.)\n\n"
        "**First, read this file** using str_replace_editor to understand the "
        "error patterns. Error Tracking aggregates similar errors together, so "
        "each issue may represent many occurrences.\n\n"
        "## Additional Context\n\n"
        f"The original Datadog query was: `{query}`\n\n"
        "If you need more details, you can use Datadog APIs via curl commands "
        "with $DD_API_KEY and $DD_APP_KEY environment variables.\n\n"
        "To search for more error tracking issues:\n"
        "```bash\n"
        f"curl -X POST '{error_tracking_url}' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -H 'DD-API-KEY: $DD_API_KEY' \\\n"
        "  -H 'DD-APPLICATION-KEY: $DD_APP_KEY' \\\n"
        '  -d \'{"data": {"attributes": {"query": "service:YOUR_SERVICE", '
        '"from": <timestamp_ms>, "to": <timestamp_ms>, "track": "logs"}, '
        '"type": "search_request"}}\'\n'
        "```\n\n"
        "To query individual log entries, use the Logs API:\n"
        "```bash\n"
        f"curl -X POST '{logs_url}' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -H 'DD-API-KEY: $DD_API_KEY' \\\n"
        "  -H 'DD-APPLICATION-KEY: $DD_APP_KEY' \\\n"
        "  -d '{\n"
        '    "filter": {\n'
        '      "query": "YOUR_QUERY_HERE",\n'
        '      "from": "now-1d",\n'
        '      "to": "now"\n'
        "    },\n"
        '    "sort": "timestamp",\n'
        '    "page": {\n'
        '      "limit": 10\n'
        "    }\n"
        "  }'\n"
        "```\n\n"
        "The Datadog query syntax supports:\n"
        "- status:error - Find error logs\n"
        "- service:my-service - Filter by service\n"
        '- "exact phrase" - Search for exact text\n'
        "- -(status:info OR status:debug) - Exclude certain statuses\n"
        "- Use time ranges to focus on recent issues\n\n"
        "The error class that I would like you to debug is characterized "
        f"by this datadog query:\n{query}\n\n"
        "To clone the GitHub repositories, use git with authentication:\n"
        "```bash\n"
        "git clone https://$GITHUB_TOKEN@github.com/OWNER/REPO.git\n"
        "```\n\n"
        "The github repos that you should clone (using GITHUB_TOKEN) are "
        f"the following:\n{repos_list}\n\n"
        "## Debugging Steps\n\n"
        "Follow these steps systematically:\n\n"
        "1. **Read the error file** - Start by reading "
        f"`{errors_file}` to understand the error patterns. "
        "Examine all examples to identify:\n"
        "   - Common error messages\n"
        "   - Stack traces and their origins\n"
        "   - Affected services\n"
        "   - Timestamps (when did errors start?)\n\n"
        "2. **Analyze the timeline** - Check when the error class started "
        "occurring/becoming frequent. Look at the timestamps in the error "
        "examples. This helps identify what code changes or deployment may "
        "have caused the issue. Code changed during the release cycle before "
        "the error occurred will be most suspicious.\n\n"
        "3. **Clone repositories** - Clone the relevant repositories using:\n"
        "   ```bash\n"
        "   git clone https://$GITHUB_TOKEN@github.com/OWNER/REPO.git\n"
        "   ```\n\n"
        "4. **Investigate the codebase** - Carefully read the code related "
        "to the error. Look at:\n"
        "   - Files mentioned in stack traces\n"
        "   - Recent commits (use git log)\n"
        "   - Related code paths\n\n"
        "5. **Develop hypotheses** - Think of 5 possible root causes and "
        "write sample code to test each hypothesis. Try to reproduce the "
        "error.\n\n"
        "6. **Create fix or summarize** - Based on your findings:\n"
        "   - If reproducible: Create a fix and optionally open a draft PR\n"
        "   - If not reproducible: Summarize your investigation, findings, "
        "and recommendations\n\n"
        "**Important**: Use the task_tracker tool to organize your work and "
        "keep track of your progress through these steps."
    )

    return prompt


def main():
    """Main function to run the Datadog debugging example."""
    parser = argparse.ArgumentParser(
        description="Debug errors from Datadog logs using OpenHands agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Datadog query to search for error logs "
        "(e.g., 'status:error service:deploy')",
    )
    parser.add_argument(
        "--repos",
        required=True,
        help="Comma-separated list of GitHub repositories to analyze "
        "(e.g., 'All-Hands-AI/OpenHands,All-Hands-AI/deploy')",
    )
    parser.add_argument(
        "--working-dir",
        default="./datadog_debug_workspace",
        help="Working directory for cloning repos and analysis "
        "(default: ./datadog_debug_workspace)",
    )

    args = parser.parse_args()

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Parse repositories
    repos = [repo.strip() for repo in args.repos.split(",")]

    # Create working directory
    working_dir = Path(args.working_dir).resolve()
    working_dir.mkdir(exist_ok=True)

    print("🔍 Starting Datadog debugging session")
    print(f"📊 Query: {args.query}")
    print(f"📁 Repositories: {', '.join(repos)}")
    print(f"🌍 Datadog site: {os.getenv('DD_SITE', 'datadoghq.com')}")
    print(f"💼 Working directory: {working_dir}")
    print()

    # Fetch error examples from Datadog
    errors_file = fetch_datadog_errors(args.query, working_dir)
    print()

    # Configure LLM
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("❌ LLM_API_KEY environment variable is required")
        sys.exit(1)

    # Get LLM configuration from environment
    model = os.getenv("LLM_MODEL", "openhands/claude-sonnet-4-5-20250929")
    base_url = os.getenv("LLM_BASE_URL")

    llm = LLM(
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
    )

    # Run debugging session
    run_debugging_session(llm, working_dir, args.query, repos, errors_file)


def run_debugging_session(
    llm: LLM,
    working_dir: Path,
    query: str,
    repos: list[str],
    errors_file: Path,
):
    """Run the debugging session with the given configuration."""
    # Register and set up tools
    register_tool("BashTool", BashTool)
    register_tool("FileEditorTool", FileEditorTool)
    register_tool("TaskTrackerTool", TaskTrackerTool)

    tools = [
        Tool(name="BashTool"),
        Tool(name="FileEditorTool"),
        Tool(name="TaskTrackerTool"),
    ]

    # Create agent
    agent = Agent(llm=llm, tools=tools)

    # Collect LLM messages for debugging
    llm_messages = []

    def conversation_callback(event: Event):
        if isinstance(event, LLMConvertibleEvent):
            llm_messages.append(event.to_llm_message())

    # Start conversation with local workspace
    conversation = Conversation(
        agent=agent, workspace=str(working_dir), callbacks=[conversation_callback]
    )

    # Send the debugging task
    debugging_prompt = create_debugging_prompt(query, repos, errors_file)

    conversation.send_message(
        message=Message(
            role="user",
            content=[TextContent(text=debugging_prompt)],
        )
    )

    print("🤖 Starting debugging analysis...")
    try:
        conversation.run()

        print("\n" + "=" * 80)
        print("🎯 Debugging session completed!")
        print(f"📁 Results saved in: {working_dir}")
        print(f"💬 Total LLM messages: {len(llm_messages)}")

        # Show summary of what was accomplished
        print("\n📋 Session Summary:")
        print("- Queried Datadog logs for error analysis")
        print("- Cloned and analyzed relevant repositories")
        print("- Investigated potential root causes")
        print("- Attempted error reproduction")

        # Check for cloned repositories
        if working_dir.exists():
            cloned_repos = [
                d for d in working_dir.iterdir() if d.is_dir() and (d / ".git").exists()
            ]
            if cloned_repos:
                print(
                    f"- Cloned repositories: {', '.join(d.name for d in cloned_repos)}"
                )
    finally:
        # Clean up conversation
        logger.info("Closing conversation...")
        conversation.close()


if __name__ == "__main__":
    main()
