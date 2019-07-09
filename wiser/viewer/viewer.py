try:
    from IPython.core.display import display, Javascript
except ModuleNotFoundError:
    raise Exception("This module requires IPython.")

import os
import ipywidgets as widgets
from traitlets import Unicode

directory = os.path.abspath(os.path.dirname(__file__))

PAGE_HTML = u"""
<div class="viewer-page" id="viewer-page-{id}-{source}"{etc}>
    {data}
</div>
"""

SOURCE_OPTION_HTML = u"""
<option value="{value}">{name}</option>
"""


class Viewer(widgets.DOMWidget):
    """Viewer for instances and associated labels."""
    _view_name           = Unicode('ViewerView').tag(sync=True)
    _view_module         = Unicode('viewer').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    html                 = Unicode('<h3>Error rendering Viewer!</h3>').tag(sync=True)
    _labels_serialized   = Unicode().tag(sync=True)

    def __init__(self, instances, height=300, additional_labels=None):
        super(Viewer, self).__init__()
        self.instances = instances
        self.n_instances = len(instances)
        self.height = height
        if additional_labels is not None:
            self.additional_labels = additional_labels
        else:
            self.additional_labels = {}

        # display js, construct html and pass on to widget model
        self.render()

    def render(self):
        """Renders viewer pane"""
        # Collects all labeling function names. We assume that all instances
        # in the collection are labeled by the same functions.
        if 'WISER_LABELS' in self.instances[0]:
            lf_names = sorted(self.instances[0]['WISER_LABELS'].keys())
        else:
            lf_names = []

        # Also sorts names of any additional label sources
        additional_names = sorted(self.additional_labels.keys())

        # Renders all pages
        pages = []
        first = True
        for i, instance in enumerate(self.instances):
            # Collects labels for instance
            labels = [instance['tags']]
            for name in additional_names:
                labels.append(self.additional_labels[name][i])
            for lf_name in lf_names:
                labels.append(instance['WISER_LABELS'][lf_name])

            for j, label_seq in enumerate(labels):
                rendered_instance = self._render_instance(instance, label_seq)
                pages.append(PAGE_HTML.format(
                    data=rendered_instance,
                    id=i,
                    source=j,
                    etc=' style="display: block;"' if first else ''
                ))
                first = False

        # Renders the label source options
        source_options = [SOURCE_OPTION_HTML.format(value=0, name="Gold Labels")]
        offset = 1
        for i, name in enumerate(additional_names):
            source_options.append(
                SOURCE_OPTION_HTML.format(value=i+offset, name=name))
        offset += len(additional_names)
        for i, lf_name in enumerate(lf_names):
            source_options.append(
                SOURCE_OPTION_HTML.format(value=i+offset, name=lf_name))

        # Render in Viewer template
        self.html = open(os.path.join(directory, 'viewer.html')).read()\
                    % (self.height, ''.join(pages), ''.join(source_options))
        display(Javascript(open(os.path.join(directory, 'viewer.js')).read()))

    def _render_instance(self, instance, labels):
        if len(instance['tokens']) != len(labels):
            raise ValueError("Number of tokens does not match number of labels")

        html = []
        active_span = False
        for i in range(len(instance['tokens'])):
            if (labels[i][0] == 'I' and not active_span) or labels[i][0] == 'B':
                if labels[i][0] == 'B':
                    html.append("</span>")
                html.append(" ")
                title = "Entity" if len(labels[i]) <= 2 else labels[i][2:]
                html.append("<span class='active' title='%s'>" % title)
                active_span = True
            elif labels[i][0] == 'O' or labels[i] == 'ABS':
                if active_span:
                    html.append("</span>")
                html.append(" ")
                active_span = False
            else:
                html.append(" ")

            html.append(instance['tokens'][i].text)

        # Closes span if still active
        if active_span:
            html.append("</span>")

        return ''.join(html)
