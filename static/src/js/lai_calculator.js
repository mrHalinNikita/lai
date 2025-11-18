odoo.define('lai_estimator.lai_calculator', function (require) {
    'use strict';

    var publicWidget = require('web.public.widget');

    publicWidget.registry.LAICalculatorWidget = publicWidget.Widget.extend({
        selector: '.lai-upload-form',

        start: function () {
            this.$el.on('submit', this._onFormSubmit.bind(this));
        },

        _onFormSubmit: function (e) {
            const $form = $(e.currentTarget);
            const $submitBtn = $form.find('button[type="submit"]');

            $submitBtn.html('<i class="fa fa-spinner fa-spin me-2"></i>Calculating...').prop('disabled', true);
        }
    });

    return publicWidget.registry.LAICalculatorWidget;
});