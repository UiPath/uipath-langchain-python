import logging


class IgnoreSpecificUrl(logging.Filter):
    def __init__(self, url_to_ignore):
        super().__init__()
        self.url_to_ignore = url_to_ignore

    def filter(self, record):
        try:
            if record.msg == 'HTTP Request: %s %s "%s %d %s"':
                # Ignore the log if the URL matches the one we want to ignore
                method = record.args[0]
                url = record.args[1]

                if method == "POST" and url.path.endswith(self.url_to_ignore):
                    # Check if the URL contains the specific path we want to ignore
                    return True
                return False

        except Exception:
            return False


def _setup_tracer_httpx_logging(url: str):
    # Create a custom logger for httpx
    # Add the custom filter to the root logger
    logging.getLogger("httpx").addFilter(IgnoreSpecificUrl(url))
