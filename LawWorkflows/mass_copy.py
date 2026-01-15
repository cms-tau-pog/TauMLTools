"""
Mass file copy utility for local and remote (XRootD) files.
Supports parallel directory exploration and file copying with progress tracking.
"""

import os
import sys
from glob import glob, escape
import fnmatch
from urllib.parse import urlparse
import concurrent.futures
from XRootD import client
from XRootD.client import FileSystem, CopyProcess
from XRootD.client.flags import StatInfoFlags
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

# -------- Utility Functions --------


def check_if_local(pattern):
    """
    Check if a given path pattern is local or remote.
    Returns True for local paths, False for remote paths (like XRootD URLs).
    Raises ValueError if pattern is ambiguous.
    """
    result = urlparse(pattern)
    if result.scheme and result.netloc:
        return False
    elif not result.scheme and not result.netloc:
        return True
    raise ValueError(f"Pattern {pattern} is neither fully remote nor local.")


def is_directory(fs, path):
    """
    Check if a remote path is a directory using XRootD FileSystem.
    Args:
        fs: XRootD FileSystem instance
        path: Remote path to check
    Returns:
        bool: True if path is a directory, False otherwise
    """
    status, info = fs.stat(path)
    if not status.ok:
        return False
    return info.flags & StatInfoFlags.IS_DIR


def ensure_directory(path):
    """
    Create directory structure if it doesn't exist.
    Works for both local and remote paths.
    For remote paths, creates all parent directories recursively.
    """
    if check_if_local(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    else:
        parsed = urlparse(path)
        client_url = f"{parsed.scheme}://{parsed.netloc}"
        fs = FileSystem(client_url)
        # Create all parent directories
        current_path = ""
        for part in parsed.path.split("/"):
            if not part:
                continue
            current_path = os.path.join(current_path, part)
            status, _ = fs.mkdir(current_path, flags=0)
            if not status.ok and status.errno != 17:  # 17 = directory exists
                print(f"Error creating directory {current_path}: {status.message}")


def get_relative_path(base_path, full_path):
    """
    Calculate relative path from base_path to full_path.
    Used to preserve directory structure when copying.
    """
    if base_path == full_path:
        return os.path.basename(base_path)
    else:
        return os.path.join(
            os.path.basename(base_path), os.path.relpath(full_path, base_path)
        )


# -------- Glob Functions --------


def recursive_glob(xrd_client, base_path, pattern_parts):
    """
    Recursively glob remote files matching pattern.
    Args:
        xrd_client: XRootD client instance
        base_path: Starting directory path
        pattern_parts: List of path components with wildcards
    Returns:
        List of matched file paths
    """
    if not pattern_parts:
        return []
    current_pattern = pattern_parts[0]
    remaining_patterns = pattern_parts[1:]
    try:
        # This check is prone to false positives, but it's better than false negatives
        if escape(current_pattern) != current_pattern:
            status, dir_list = xrd_client.dirlist(base_path)
            if not status.ok:
                print(f"Error listing directory {base_path}: {status.message}")
                return []
            matching_entries = fnmatch.filter(
                [entry.name for entry in dir_list.dirlist], current_pattern
            )
        else:
            matching_entries = [current_pattern]
        if not remaining_patterns:
            return [os.path.join(base_path, match) for match in matching_entries]
        matched_paths = []
        for match in matching_entries:
            full_match_path = os.path.join(base_path, match)
            matched_paths.extend(
                recursive_glob(xrd_client, full_match_path, remaining_patterns)
            )
        return matched_paths
    except Exception as e:
        print(f"Error processing {base_path} with pattern {current_pattern}: {e}")
        return []


def remote_glob(pattern):
    """
    Universal glob function that works for both local and remote paths.
    For remote paths, uses XRootD to list and match files.
    """
    if check_if_local(pattern):
        return glob(pattern)
    parsed_url = urlparse(pattern)
    client_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    file_path = parsed_url.path
    path_parts = file_path.strip("/").split("/")
    xrd_client = client.FileSystem(client_url)
    matched_files = recursive_glob(xrd_client, "/", path_parts)
    return [f"{client_url}/{file_}" for file_ in matched_files]


# -------- Copy Functions --------


def xrdcp_copy(source, destination):
    """
    Copy single file using XRootD CopyProcess.
    Handles both local-to-remote and remote-to-local transfers.
    """
    try:
        # Ensure destination has proper format
        if check_if_local(destination):
            # Local paths need file:// prefix for XRootD
            formatted_dest = f"file://{os.path.abspath(destination)}"
        else:
            formatted_dest = destination
        cp = CopyProcess()
        cp.add_job(source, formatted_dest)
        status = cp.prepare()
        if not status.ok:
            print(f"Error preparing copy {source} -> {destination}: {status.message}")
            return
        status = cp.run()
        if not status[0].ok:
            print(f"Error copying {source} -> {destination}: {status[0].message}")
    except Exception as e:
        print(f"Exception while copying {source} -> {destination}: {e}")


def explore_remote_directory(client_url, base_path, current_path):
    """
    Worker function for parallel directory exploration.
    Lists contents of one directory, identifying files and subdirectories.
    Args:
        client_url: XRootD server URL
        base_path: Original source path for relative path calculation
        current_path: Directory to explore
    Returns:
        List of tuples: (type, (base_path, full_path))
        where type is either "file" or "dir"
    """
    try:
        files = []
        fs = FileSystem(client_url)
        status, listing = fs.dirlist(current_path)
        if not status.ok:
            return files
        for entry in listing.dirlist:
            entry_path = os.path.join(current_path, entry.name)
            full_path = f"{client_url}{entry_path}"
            if is_directory(fs, entry_path):
                files.append(("dir", (base_path, full_path)))
            else:
                files.append(("file", (base_path, full_path)))
        return files
    except Exception as e:
        print(f"Error exploring {current_path}: {e}")
        return []


def list_all_files(paths, max_workers=4):
    """
    Parallel recursive file listing using thread pool.
    Uses breadth-first exploration to handle large directory structures.
    Args:
        paths: List of source paths to explore
        max_workers: Maximum number of parallel threads
    Returns:
        List of tuples: (base_path, full_path) for all files
    """
    all_files = []
    dirs_to_explore = deque()
    # Initial population from glob results
    for matched_path in paths:
        parsed = urlparse(matched_path)
        client_url = f"{parsed.scheme}://{parsed.netloc}"
        fs = FileSystem(client_url)
        if is_directory(fs, parsed.path):
            # For directories, use matched_path as base for all files within
            dirs_to_explore.append((client_url, matched_path, parsed.path))
        else:
            # For direct file matches, use parent dir as base
            all_files.append((matched_path, matched_path))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        while dirs_to_explore and len(futures) < max_workers:
            client_url, base_path, current_path = dirs_to_explore.popleft()
            future = executor.submit(
                explore_remote_directory, client_url, base_path, current_path
            )
            futures.add(future)
        while futures:
            done, futures = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                try:
                    results = future.result()
                    for type_, (base, full_path) in results:
                        if type_ == "dir":
                            client_url = (
                                urlparse(full_path).scheme
                                + "://"
                                + urlparse(full_path).netloc
                            )
                            path = urlparse(full_path).path
                            if dirs_to_explore or len(futures) < max_workers:
                                dirs_to_explore.append((client_url, base, path))
                            else:
                                sub_results = explore_remote_directory(
                                    client_url, base, path
                                )
                                for _, item in sub_results:
                                    if _ == "file":
                                        all_files.append(item)
                        else:
                            all_files.append((base, full_path))
                except Exception as e:
                    print(f"Error processing directory: {e}")
            while dirs_to_explore and len(futures) < max_workers:
                client_url, base_path, current_path = dirs_to_explore.popleft()
                future = executor.submit(
                    explore_remote_directory, client_url, base_path, current_path
                )
                futures.add(future)
    return all_files


def mass_copy(sources, destination, max_workers=4, verbose=True):
    """
    Main function for parallel mass file copying.
    1. Discovers source files using glob patterns
    2. Lists all files in directories recursively
    3. Copies files in parallel with progress tracking
    Args:
        sources: Single path or list of source paths (with wildcards)
        destination: Target directory
        max_workers: Maximum number of parallel copy operations
        verbose: Whether to show progress bars and status messages
    """
    max_workers = int(max_workers)
    if isinstance(sources, str):
        sources = [sources]
    try:
        current_width = os.get_terminal_size().columns
    except OSError:
        current_width = 140
    console = Console(width=current_width)
    # Discovery phase
    if verbose:
        console.print("[yellow]Discovering files...", end="")
    globbed_paths = []
    for path in sources:
        globbed_paths.extend(remote_glob(path))
    if verbose:
        console.print("[green] Done!")
        # Listing phase
        console.print("[yellow]Listing directories...", end="")
    files_to_copy = list_all_files(globbed_paths, max_workers=max_workers)
    if verbose:
        console.print(f"[green] Done! Found {len(files_to_copy)} files.")
    # Copy phase with progress
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        expand=True,
    )
    with progress:
        task_copy = progress.add_task("[green]Copying files", total=len(files_to_copy))
        ensure_directory(destination)

        def copy_with_progress(dat):
            root, src = dat
            rel_path = get_relative_path(root, src)
            dest = os.path.join(destination, rel_path)
            ensure_directory(os.path.dirname(dest))
            current_progress = progress.tasks[task_copy].completed
            if verbose:
                progress.update(
                    task_copy,
                    description=f"[green]Copying [{current_progress + 1}/{len(files_to_copy)}]: {os.path.basename(src)}",
                )
            xrdcp_copy(src, dest)
            if verbose:
                progress.advance(task_copy)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(copy_with_progress, files_to_copy)
    console.print("[green] Done!")


if __name__ == "__main__":
    mass_copy(*sys.argv[1:])
