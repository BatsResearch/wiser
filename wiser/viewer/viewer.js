require.undef('viewer');

// NOTE: all elements should be selected using this.$el.find to avoid collisions with other Viewers

define('viewer', ["@jupyter-widgets/base"], function(widgets) {
    var ViewerView = widgets.DOMWidgetView.extend({
        // Main rendering function
        render: function() {
            // Insert the html
            this.$el.append(this.model.get('html'));
            this.nPages = this.model.get('n_instances');
            this.$el.append(this.nPages);
            // Set the instance id
            this.id  = 0;
            // Set the label source
            this.source = 0;

            // Enable buttons for changing page
            var that = this;
            this.$el.find("#next-page").click(function() {
                that.switchPage(1);
            });
            this.$el.find("#prev-page").click(function() {
                that.switchPage(-1);
            });

            // Enable select menu for changing label source
            this.$el.find("#source").change(function() {
                that.switchSource();
            })
        },

        switchPage: function(inc) {
            this.$el.find(".viewer-page").hide();
            if (this.id + inc < 0) {
                this.id = 0;
            } else if (this.id + inc >= this.nPages - 1) {
                this.id = this.nPages - 1;
            } else {
                this.id += inc;
            }
            this.$el.find("#viewer-page-"+this.id+"-"+this.source).show();

            // Show page id
            this.$el.find("#page").html(this.id);
        },

        switchSource: function() {
            this.source = this.$el.find("#source").val();
            this.switchPage(0);
        }
    });

    return {
        ViewerView: ViewerView
    };
});