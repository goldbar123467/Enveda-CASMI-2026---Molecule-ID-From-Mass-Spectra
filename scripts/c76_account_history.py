"""Read all scoring-history pages, retaining opaque server pagination tokens."""
from c09_submission_guard import field
from c76_submission_contract import COMP


def complete_history(api):
    from kaggle.api.kaggle_api_extended import ApiListSubmissionsRequest
    records = []; references = set(); tokens = {''}; token = ''
    with api.build_kaggle_client() as client:
        for _ in range(1000):
            request = ApiListSubmissionsRequest()
            request.competition_name = COMP; request.page_size = 100; request.page_token = token
            response = client.competitions.competition_api_client.list_submissions(request)
            for row in response.submissions or []:
                assert row is not None
                reference = str(field(row, 'ref', ''))
                assert reference.isdigit() and int(reference) > 0, 'Unrecognized account reference'
                assert reference not in references, 'Repeated history row; retry observation without submitting'
                references.add(reference); records.append(row)
            token = response.next_page_token
            if not token: return records
            assert token not in tokens, 'Repeated history token; no scoring release'
            tokens.add(token)
    raise RuntimeError('Account history pagination exceeded the bound; no scoring release')
