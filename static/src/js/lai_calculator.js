/** @odoo-module alias=lai_estimator.lai_calculator **/
$(document).ready(function () {

    $('.lai-upload-form').on('submit', function () {
        const $submitBtn = $(this).find('button[type="submit"]');
        $submitBtn.html('<i class="fa fa-spinner fa-spin me-2"></i>Calculating...').prop('disabled', true);
    });
});