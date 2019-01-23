from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, SelectField, DecimalField, IntegerField
from netfui.wtforms_extended_selectfield import ExtendedSelectField
from flask_wtf.file import FileField, FileAllowed
from wtforms.validators import DataRequired,NumberRange

class TrainForm(FlaskForm):
    
    project = SelectField('Project',choices=[])
    experiment = StringField('Experiments',validators=[DataRequired()])

    dataset = SelectField('Dataset',choices=[])
    datasetparam = StringField('Dataset Parameters')
    model = SelectField('Model',choices=[])
    modelparam = StringField('Model Parameters')
    optimizer = SelectField('Optimizer',choices=[('SGD','SGD'),('RMSprop','RMSprop'),('Adam','Adam')])
    optimizerparam = StringField('Optimizer Parameters')
    lrschedule = SelectField('LR Schedule',choices=[('none','none'),('rop','rop'),('step','step')])
    loss = SelectField('Loss',choices=[])
    lossparam = StringField('Loss Parameters')
    
    epochs = IntegerField('Epochs',validators=[DataRequired(),NumberRange(1,1000000)])
    batch_size = IntegerField('Batch Size',validators=[DataRequired(),NumberRange(1,10000000)])
    visdom = BooleanField('Use visdom')
    show_rate = IntegerField('Plot Rate',validators=[DataRequired(),NumberRange(0,1000)])
    print_rate = IntegerField('Print Rate',validators=[DataRequired(),NumberRange(0,1000)])
    save_rate = IntegerField('Model save Rate',validators=[DataRequired(),NumberRange(0,1000)])
    use_cuda = SelectField('GPU',choices=[])
    parallel = BooleanField('Use multi-GPU')
    train_worker = IntegerField('Train workers',validators=[DataRequired(),NumberRange(0,20)])
    test_worker = IntegerField('Test workers',validators=[DataRequired(),NumberRange(0,20)])
    resume = BooleanField('Resume on fail')

    submit = SubmitField('Add experiment')

class TestForm(FlaskForm):
    
    project = SelectField('Project',choices=[])
    experiment = StringField('Experiments',validators=[DataRequired()])

    model = ExtendedSelectField('Model',choices=())
    modelarg = StringField('Model argument name')
    
    inputs  = StringField('Input path')
    inputsarg = StringField('Input argument name')

    otherarg = StringField('Other arguments')
    use_cuda = SelectField('GPU',choices=[])

    submit = SubmitField('Add test experiment')

class ProjectForm(FlaskForm):
    name = StringField('Project name:',validators=[DataRequired()])
    path = StringField('Train path:',validators=[DataRequired()])
    exe = StringField('Train executable:',validators=[DataRequired()])

    test_path = StringField('Test path (Optional):')
    test_exe = StringField('Test executable (Optional):')

    submit = SubmitField('Add project')


