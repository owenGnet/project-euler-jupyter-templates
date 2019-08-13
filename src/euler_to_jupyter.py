import argparse
import copy
import json
import logging
import time

from collections import deque
from pathlib import Path

import numpy as np
import requests
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup

TIMEOUT_SECONDS = 5
LOCAL_IMAGES_DIR = Path('./problems/images')

logger = logging.getLogger()


def prep_local_env(ipynb_template_file):
    """Confirm template file exists and create directory for downloading images"""
    if ipynb_template_file.is_file():
        if not LOCAL_IMAGES_DIR.exists():
            logger.info(f'creating local "{LOCAL_IMAGES_DIR}" directory for downloaded images')
            LOCAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    else:
        raise FileNotFoundError('template file path "{}" does not appear to be valid'.format(ipynb_template_file))


def get_template_data(jup_path):
    """Return dict of first cell + metadata/versioning info from local template .ipynb"""
    try:
        with open(jup_path, 'r', encoding='utf8') as f:
            logger.info(f'opening {jup_path} to retrieve template data')
            jup_data = json.load(f)
            first_cell = jup_data['cells'][0]

    except json.decoder.JSONDecodeError as e:
        logger.error(f'Jupyter template file does not appear to be valid JSON: "{e}"')
        raise
    except KeyError as e:
        logger.error(f'Unable to retrieve first cell from Jupyter template file: "{e}"')
        raise

    template_keys = ['metadata', 'nbformat', 'nbformat_minor', ]
    # TODO: insert tippy-top cell that embeds/references the euler logo image
    template_data = {'cells': [first_cell, ]}
    missing_keys = set(template_keys).difference(jup_data.keys())
    if missing_keys:
        raise KeyError(f'Jupyter template file missing one or more expected keys: {missing_keys}')
    template_data.update({k: v for k, v in jup_data.items() if k in template_keys})

    return template_data


def get_the_response(url):
    """GET the problem web page"""
    response = None
    try:
        response = requests.get(url, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except HTTPError as http_err:
        logger.error(f'Failed to download {url}, HTTP response: {http_err}')
    except Exception as e:
        logger.error(f'Failed to download {url}: {e}')

    return response


def download_images(imgs):
    """Save any images on page to local directory"""
    had_download_issue = False
    for img in imgs:
        image_url = 'https://projecteuler.net/{}'.format(img.get('src'))
        logger.info(f'downloading image {image_url}')
        image_name = Path(image_url).name
        image = get_the_response(image_url)
        if image:
            (LOCAL_IMAGES_DIR / image_name).write_bytes(image.content)
        else:
            had_download_issue = True

    return not had_download_issue


def parse_input(raw_pnums):
    """Parse the (string) of problem number references into problem numbers to retreive"""
    # undocumented "rand,#" format intended primarily for testing
    if raw_pnums.startswith('rand,'):
        count = int(raw_pnums.split(',')[1])
        max_pnum = 677  # max as of Aug, 2019
        try:
            recent_problems = requests.get('https://projecteuler.net/recent')
            soup = BeautifulSoup(recent_problems.text, 'html.parser')
            first_cell = soup.find('table', id='problems_table').find('td').text
            max_pnum = int(first_cell)
            logger.info(f'Found latest problem number on /recent page: {max_pnum}')
        except Exception as e:
            logger.error(f'Error determining last published problem number, set as {max_pnum}. Exception: "{e}"')
        problem_numbers = np.random.choice(range(1, (max_pnum + 1)), count, replace=False)
    else:
        # problem_numbers = [int(n) for n in pnums.split(',')]
        nums = set()
        csv_parts = raw_pnums.split(',')
        try:
            for part in csv_parts:
                if '-' in part:
                    begin, end = part.split('-')
                    nums.update(range(int(begin), int(end) + 1))
                else:
                    nums.add(int(part))
        except ValueError as e:
            logger.error(f'Unable to parse problem numbers value "{raw_pnums}", exception: {e}')
            raise
        problem_numbers = sorted(list(nums))

    return problem_numbers


def populate_problems(problem_numbers):
    """Wrap the web page scrape/image downloading, return dict: problem_number: raw_content"""
    try:
        pnums = parse_input(problem_numbers)
    except ValueError as e:
        logger.error(f'Unable to parse problem numbers from input: {e}')
        raise e

    logger.info('{} problems to fetch: {}'.format(len(pnums), pnums))
    base_url = 'https://projecteuler.net/problem='
    problems_with_issues = set()

    problem_contents = {}
    for pnum in pnums:
        problem_url = f'{base_url}{pnum}'
        logger.info(f'requesting {problem_url}')

        page = get_the_response(problem_url)
        if not page:
            logging.warning('uh oh: {}'.format(pnum))
            problems_with_issues.add(pnum)
            continue

        soup = BeautifulSoup(page.text, 'html.parser')
        content = soup.find('div', id='content')
        embedded_images = content.find('div', 'problem_content').find_all('img')
        full_image_success = download_images(embedded_images)
        if not full_image_success:
            problems_with_issues.add(pnum)
        problem_contents[pnum] = {'content_raw': content}
        time.sleep(1)

    if problems_with_issues:
        issues_pretty = ','.join(str(p) for p in sorted(problems_with_issues))
        logger.warning(f'Issues encountered during the download of following problems: {issues_pretty}')
        logger.warning('Check ERROR lines in log for further details')

    return problem_contents


def process_problems(problems):
    """Inject styles, update image pathing into raw_content, produce final_content"""
    for pnum, contents in problems.items():
        raw_content = contents['content_raw']
        problem_description = raw_content.find('h2')
        problem_description['style'] = 'color: #6b4e3d;'
        problem_info = raw_content.find('div', id='problem_info')
        problem_info['style'] = 'font-family: Consolas;'
        problem_content = raw_content.find('div', 'problem_content')
        problem_content['style'] = ('background-color: #fff; color: #111; padding: 20px;'
                                    'font-family: "Segoe UI", Arial, sans-serif; font-size: 110%;'
                                    'border: solid 1px #bbb; box-shadow: 5px 5px 5px #bbb;')
        problem_html = '{}\n{}\n{}'.format(problem_description, problem_info, problem_content)
        problem_html = problem_html.replace('src=\"project/images/', '<img src="./images/')
        problems[pnum]['content_final'] = problem_html

    return problems


def populate_cell_pair_q(problem_numbers, include_answer=False):
    """Wrap process of getting html/images, populate a deque of problem/empty-cell pairs"""
    raw_problems = populate_problems(problem_numbers)
    enhanced_problems = process_problems(raw_problems)

    q = deque()
    for pnum, v in enhanced_problems.items():
        problem_cell_pair = assemble_cell_pair(pnum, v['content_final'], include_answer)
        q.append(problem_cell_pair)

    return q


def assemble_cell_pair(pnum, source, include_answer=False):
    """Take problem html, pair with empty "answer" cell, and return dict formatted for Jupyter"""
    problem_cell = {
        'cell_type': 'markdown',
        'metadata': {
            'collapsed': True
        },
    }

    empty_cell_contents = ['42', ] if include_answer else []
    empty_code_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {
            'collapsed': True
        },
        'outputs': [],
        'source': empty_cell_contents
    }

    logger.info(f'Assembling cell pair for problem {pnum}')
    problem_cells = []
    one_problem_cell = {k: v for k, v in problem_cell.items()}
    one_problem_cell['source'] = source
    problem_cells.append(one_problem_cell)
    problem_cells.append(empty_code_cell)

    return problem_cells


# TODO: support for passing problems_per_notebook from command line
def gather_notebooks(template_data, cell_pairs_q, problems_per_notebook=10):
    """Create complete (dict) Jupyter notebooks. limit number of problems in each notebook"""
    range_val = (len(cell_pairs_q) // problems_per_notebook) + 1
    for i in range(range_val):
        notebook_cells = []
        full_data = copy.deepcopy(template_data)
        for j in range(problems_per_notebook):
            try:
                notebook_cells.extend(cell_pairs_q.popleft())
            except IndexError:
                break
        if not notebook_cells:
            continue
        full_data['cells'].extend(notebook_cells)
        logging.info(f'notebook contents with {len(notebook_cells)/2:.0f} problems assembled')
        yield full_data


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--template-path',
        type=str,
        help='local path to template Jupyter notebook, first cell will be used for templates',
        required=True)
    parser.add_argument(
        '--problem-numbers',
        type=str,
        help='parseable string of problem numbers and/or range of numbers',
        required=True)
    parser.add_argument(
        '--debug',
        type=bool,
        default=False)
    parser.add_argument(
        '--include-answer',
        type=bool,
        default=False)
    arguments = parser.parse_args()

    return arguments


def main():
    """Entry point to cover full workflow"""
    args = parse_args()

    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    stream_handler.setFormatter(formatter)
    if args.debug:
        stream_handler.setLevel(logging.DEBUG)
    else:
        stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    template_path = Path(args.template_path)
    prep_local_env(template_path)
    template_data = get_template_data(template_path)
    cell_pairs_q = populate_cell_pair_q(args.problem_numbers, args.include_answer)

    # TODO: obtain pBEGIN & pEND for each noteboook, for ipynb file naming
    notebook_contents = gather_notebooks(template_data, cell_pairs_q)
    out_path = './problems/euler_{}.ipynb'
    for idx, content in enumerate(notebook_contents):
        notebook_path = Path(out_path.format(idx + 1))
        logging.info(f'Writing notebook {idx+1}: {notebook_path.name}')
        with open(notebook_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(content))

if __name__ == '__main__':
    main()
