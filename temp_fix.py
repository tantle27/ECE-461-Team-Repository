import re

# Read handlers.py
with open('src/handlers.py', 'r') as f:
    content = f.read()

# Fix all long lines with rate limiting messages
content = re.sub(
    r'ctx\.fetch_logs\.append\(f"Rate limited, retrying in \{retry_delay\}s\.\.\."\)',
    'msg = f"Rate limited, retrying in {retry_delay}s..."\n                    ctx.fetch_logs.append(msg)',
    content
)

# Fix GitHub API URL line
content = re.sub(
    r'response = requests\.get\(f"https://api\.github\.com/repos/\{owner\}/\{repo\}"\)',
    'url = f"https://api.github.com/repos/{owner}/{repo}"\n                response = requests.get(url)',
    content
)

# Fix GitHub error message
content = re.sub(
    r'raise Exception\(f"GitHub API error: \{response\.status_code\}"\)',
    'error_msg = f"GitHub API error: {response.status_code}"\n                    raise Exception(error_msg)',
    content
)

# Write back
with open('src/handlers.py', 'w') as f:
    f.write(content)

print('Fixed handlers.py')
