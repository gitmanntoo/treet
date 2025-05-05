import concurrent.futures
from dataclasses import dataclass, field
import datetime
import grp
import json
from pathlib import Path
import pwd
import stat
import threading
from typing import Annotated, Optional

from magika import Magika
from magika.types import MagikaResult
import pandas as pd
from rich.console import Console
from rich.progress import Progress
from rich.tree import Tree
import typer
from typer import Typer
import yaml

console = Console()
app = Typer()
mgk = Magika()


@dataclass
class SimpleProgress:
    """
    Simple progress bar for console output.
    """
    name: str
    total: int = 1_000_000
    # Internal progress bar
    _progress: Progress = None
    _task: Progress = None
    _total_label: int = 0

    def _new_task(self):
        self._total_label += self.total
        self._task = self._progress.add_task(
            f'{self.name} {self._total_label:,d}',
            total=self.total,
        )

    def __enter__(self):
        self._progress = Progress()
        self._progress.start()
        self._new_task()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._progress.stop()

    def update(self, n: int = 1):
        if self._progress.tasks[self._task].completed + n > self.total:
            self._new_task()
        self._progress.update(self._task, advance=n)


@dataclass
class Config:
    """
    Configuration class for the Treet application.
    """
    # General settings
    root: Path
    stat: bool = False
    magika: bool = False
    pretty: bool = False
    output: Optional[Path] = None
    tree: int = None


@dataclass
class FileInfoOutputOptions:
    relative_to: Optional[Path] = None
    do_sort: bool = True
    pretty: bool = False


@dataclass
class FileInfo:
    """
    Class to hold information about a file.
    """
    path: Path
    depth: int = 0
    # Fields from os.stat_result
    st_size: int = None
    st_mtime: float = None
    st_ctime: float = None
    st_atime: float = None
    st_mode: int = None
    st_uid: int = None
    st_gid: int = None
    st_nlink: int = None
    st_dev: int = None
    st_ino: int = None
    # Fields from magika
    magika_group: str = None
    magika_label: str = None
    magika_mime_type: str = None
    # Capture errors by function name.
    errors: dict = field(default_factory=dict)
    # Lock for thread-safe access
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __lt__(self, other: 'FileInfo') -> bool:
        """
        Compare two FileInfo objects based on their path.
        """
        self_posix = self.path.as_posix()
        self_posix_lower = self_posix.lower()
        other_posix = other.path.as_posix()
        other_posix_lower = other_posix.lower()

        if self_posix_lower == other_posix_lower:
            # Use case-sensitive comparison if lowercase paths are equal
            return self_posix < other_posix
        else:
            return self_posix_lower < other_posix_lower

    def is_dir(self) -> bool:
        """
        Check if the path is a directory ignoring errors.
        """
        try:
            return self.path.is_dir()
        except OSError as e:
            self.errors['is_dir'] = e

        return False

    def iterdir(self):
        """
        Iterate over the directory contents handling errors.
        """
        try:
            for item in self.path.iterdir():
                yield item
        except OSError as e:
            self.errors['iterdir'] = e

    def update_stat(self) -> 'FileInfo':
        """
        Update the file information using os.stat.
        """
        try:
            stat_info = self.path.stat()

            with self._lock:
                self.st_size = stat_info.st_size
                self.st_mtime = stat_info.st_mtime
                self.st_atime = stat_info.st_atime
                self.st_mode = stat_info.st_mode
                self.st_uid = stat_info.st_uid
                self.st_gid = stat_info.st_gid

                # These fields are not always available on all platforms
                try:
                    self.st_ctime = stat_info.st_ctime
                except AttributeError:
                    self.st_ctime = None
                try:
                    self.st_birthtime = stat_info.st_birthtime
                except AttributeError:
                    self.st_birthtime = None
        except OSError as e:
            with self._lock:
                self.errors['update_stat'] = e

        return self

    def update_magika(self) -> 'FileInfo':
        """
        Update the file information using magika.
        """
        try:
            result: MagikaResult = mgk.identify_path(self.path)
            if result is None:
                raise ValueError("Magika returned None")

            with self._lock:
                self.magika_group = result.output.group
                self.magika_label = str(result.output.label)
                self.magika_mime_type = result.output.mime_type
        except Exception as e:
            with self._lock:
                self.errors['update_magika'] = e

        return self

    def relative_path(self, relative_to: Path = None) -> Path:
        """
        Get the relative path of the file ignoring errors.
        """
        if relative_to is None:
            return self.path

        try:
            return self.path.relative_to(relative_to)
        except ValueError:
            return self.path

    def pretty_size(self, size: int) -> str:
        """
        Convert a size in bytes to a human-readable format.
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                if unit == 'B':
                    # Avoid showing decimal places for bytes
                    return f"{size} {unit}"
                return f"{size:.3f} {unit}"
            size /= 1024
        return f"{size:,.3f} PB"

    def pretty_time(self, timestamp: float) -> datetime.datetime:
        """
        Convert a timestamp to a human-readable format.
        """
        if timestamp is None:
            return None
        return datetime.datetime.fromtimestamp(timestamp).isoformat()

    def pretty_mode(self, mode: int) -> str:
        """
        Convert a file mode to a human-readable format.
        """
        return stat.filemode(mode)

    def pretty_uid(self, uid: int) -> str:
        """
        Convert a user ID to a human-readable format.
        """
        try:
            return pwd.getpwuid(uid).pw_name
        except Exception as e:
            self.errors['pretty_uid'] = e
            return str(uid)

    def pretty_gid(self, gid: int) -> str:
        """
        Convert a group ID to a human-readable format.
        """
        try:
            return grp.getgrgid(gid).gr_name
        except Exception as e:
            self.errors['pretty_gid'] = e
            return str(gid)

    def to_dict(self, opt: FileInfoOutputOptions) -> dict:
        """
        Convert the FileInfo object to a dictionary with Python types.
        """
        output = {
            'path': self.relative_path(opt.relative_to),
            'depth': self.depth,
        }

        if self.st_size is not None:
            output['st_size'] = self.st_size
            output['st_mtime'] = self.st_mtime
            output['st_ctime'] = self.st_ctime
            output['st_birthtime'] = self.st_birthtime
            output['st_atime'] = self.st_atime
            output['st_mode'] = self.st_mode
            output['st_uid'] = self.st_uid
            output['st_gid'] = self.st_gid

            if opt.pretty:
                output['pretty_size'] = self.pretty_size(self.st_size)
                output['pretty_mtime'] = self.pretty_time(self.st_mtime)
                output['pretty_ctime'] = self.pretty_time(self.st_ctime)
                output['pretty_birthtime'] = self.pretty_time(self.st_birthtime)
                output['pretty_atime'] = self.pretty_time(self.st_atime)
                output['pretty_mode'] = self.pretty_mode(self.st_mode)
                output['pretty_uid'] = self.pretty_uid(self.st_uid)
                output['pretty_gid'] = self.pretty_gid(self.st_gid)

        if self.magika_group is not None:
            output['magika_group'] = self.magika_group
            output['magika_label'] = self.magika_label
            output['magika_mime_type'] = self.magika_mime_type

        if self.errors:
            output['errors'] = self.errors

        return output

    def to_json_dict(self, opt: FileInfoOutputOptions) -> dict:
        """
        Convert the FileInfo object to a dictionary with JSON-serializable types.
        """
        output = self.to_dict(opt)

        # Convert Python types to JSON-serializable types
        output['path'] = str(output['path'])

        if 'errors' in output:
            output['errors'] = {
                key: str(value)
                for key, value in output['errors'].items()
            }

        return output

    def tree_label(self) -> str:
        """
        Generate a label for the tree representation.
        """
        label = f"{self.path.name}"
        if self.is_dir():
            label += "/"
            return label

        if self.st_size is not None:
            label += f" {self.pretty_size(self.st_size)}"
        if self.magika_group is not None:
            label += f" {self.magika_group}/{self.magika_label}"
        return label


def scan_path(config: Config, path: Path, depth: int = 0) -> FileInfo:
    """
    Scan the directory tree and gather file information.
    """
    info = FileInfo(path=path, depth=depth)
    if config.stat:
        info.update_stat()
    yield info

    if not info.is_dir():
        return

    for item in info.iterdir():
        yield from scan_path(config, item, depth=depth + 1)


def concurrent_map(func, iterable):
    futures: list[concurrent.futures.Future] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for item in iterable:
            f = executor.submit(func, item)
            futures.append(f)

        while futures:
            not_done = []
            for f in futures:
                try:
                    yield f.result(timeout=0.1)
                except concurrent.futures.TimeoutError:
                    not_done.append(f)

            futures = not_done


def do_scan(config: Config) -> list[FileInfo]:
    start_time = datetime.datetime.now()
    scan_results = []

    with SimpleProgress("Scanning", total=1_000_000) as progress:
        for file_info in scan_path(config, config.root):
            scan_results.append(file_info)
            progress.update()

    end_time = datetime.datetime.now()
    console.print(
        f"Scanned {len(scan_results):,d} files in "
        f"{end_time - start_time}"
    )

    return scan_results


def do_magika(config: Config, scan_results: list[FileInfo]) -> None:
    start_time = datetime.datetime.now()

    with SimpleProgress("Magika", total=len(scan_results)) as progress:
        for _ in concurrent_map(
            lambda x: x.update_magika(),
            scan_results,
        ):
            progress.update()

    end_time = datetime.datetime.now()
    console.print(
        f"Magika {len(scan_results):,d} files in "
        f"{end_time - start_time}"
    )


def output_json(
    config: Config, scan_results: list[FileInfo], opt: FileInfoOutputOptions
):
    with config.output.open('w') as f:
        json.dump(
            [file_info.to_json_dict(opt) for file_info in scan_results],
            f,
        )


def output_jsonl(
    config: Config, scan_results: list[FileInfo], opt: FileInfoOutputOptions
):
    with config.output.open('w') as f:
        for file_info in scan_results:
            f.write(json.dumps(file_info.to_json_dict(opt)) + '\n')


def output_yaml(
    config: Config, scan_results: list[FileInfo], opt: FileInfoOutputOptions
):
    with config.output.open('w') as f:
        yaml.dump(
            [file_info.to_json_dict(opt) for file_info in scan_results],
            f,
        )


def output_csv(
    config: Config, scan_results: list[FileInfo], opt: FileInfoOutputOptions
):
    df = pd.DataFrame(
        [file_info.to_dict(opt) for file_info in scan_results]
    )
    df.to_csv(config.output, index=False)


def output_xlsx(
    config: Config, scan_results: list[FileInfo], opt: FileInfoOutputOptions
):
    df = pd.DataFrame(
        [file_info.to_dict(opt) for file_info in scan_results]
    )
    df.to_excel(config.output, index=False)


OUTPUT_HANDLERS = {
    '.json': output_json,
    '.jsonl': output_jsonl,
    '.yaml': output_yaml,
    '.csv': output_csv,
    '.xlsx': output_xlsx,
}


def do_output(
    config: Config, 
    scan_results: list[FileInfo],
    do_sort: bool = True,
    relative_paths: bool = True,
    pretty: bool = False,
) -> None:
    start_time = datetime.datetime.now()

    config.output.parent.mkdir(parents=True, exist_ok=True)

    opt = FileInfoOutputOptions(
        relative_to=config.root if relative_paths else None,
        do_sort=do_sort,
        pretty=pretty,
    )

    if opt.do_sort:
        scan_results.sort()

    # Execute the appropriate output handler based on the file suffix
    if config.output.suffix not in OUTPUT_HANDLERS:
        raise ValueError(
            f"Unsupported output format: {config.output.suffix}"
        )
    else:
        OUTPUT_HANDLERS[config.output.suffix](
            config, scan_results, opt,
        )

    end_time = datetime.datetime.now()
    console.print(
        f"Wrote {len(scan_results):,d} files to "
        f"{config.output} in {end_time - start_time}"
    )


def do_tree(
    config: Config, scan_results: list[FileInfo]
) -> None:
    """
    Print the directory tree to the console.
    """

    # Sort the scan results first.
    scan_results.sort()

    tree = Tree(config.root.as_posix())

    # Keep track of the tree nodes for directories.
    tree_nodes = {
        config.root: tree,
    }

    for file_info in scan_results:
        if file_info.depth > config.tree:
            continue

        if file_info.is_dir():
            if file_info.path not in tree_nodes:
                # Add a new node for the directory
                parent = file_info.path.parent
                node = tree_nodes[parent].add(
                    file_info.tree_label())
                tree_nodes[file_info.path] = node
        else:
            # Add a node for the file
            parent = file_info.path.parent
            tree_nodes[parent].add(
                file_info.tree_label())

    console.print(tree)


@app.command()
def main(
    root: Annotated[
        Path, typer.Argument(help="Root directory to scan")],
    stat: Annotated[
        bool, typer.Option(help="Run os.stat on each file")] = False,
    magika: Annotated[
        bool, typer.Option(help="Run magika on each file")] = False,
    pretty: Annotated[
        bool, typer.Option(help="Convert output to prettier formats")] = False,
    output: Annotated[
        Path, typer.Option(help="Write output to a file")] = None,
    tree: Annotated[
        int, typer.Option(help="Output a tree with specified depth")] = None,
):
    """
    Gather file information in a directory tree.
    """
    config = Config(
        root=root,
        stat=stat,
        magika=magika,
        pretty=pretty,
        output=output,
        tree=tree,
    )

    if config.output and config.output.suffix not in OUTPUT_HANDLERS:
        raise ValueError(
            f"Unsupported output format: {config.output.suffix}"
        )

    scan_results = do_scan(config)

    if config.magika:
        do_magika(config, scan_results)

    if config.output:
        do_output(
            config, scan_results,
            do_sort=True,
            relative_paths=True,
            pretty=config.pretty,
        )

    if config.tree is not None:
        do_tree(config, scan_results)


if __name__ == "__main__":
    # Run the Typer app
    typer.run(main)
