import argparse
from skeleton_to_human.data.image_folder import default_loader
import pytorch_lightning as pl
import torch
from skeleton_to_human.models import networks

OPTIMIZER = "Adam"
LR = 2e-4
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100
BETA_1 = 0.5


class Accuracy(pl.metrics.Accuracy):
    """Accuracy Metric with a hack."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Metrics in Pytorch-lightning 1.2+ versions expect preds to be between 0 and 1 else fails with the ValueError:
        "The `preds` should be probabilities, but values were detected outside of [0,1] range."
        This is being tracked as a bug in https://github.com/PyTorchLightning/metrics/issues/60.
        This method just hacks around it by normalizing preds before passing it in.
        Normalized preds are not necessary for accuracy computation as we just care about argmax().
        """
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.nn.functional.softmax(preds, dim=-1)
        super().update(preds=preds, target=target)


class BaseLitModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        # Todo: use self.hparams.abc everywhere
        self.save_hyperparameters(args)
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)
        self.beta1 = self.args.get("beta1", BETA_1)
        self.niter_fix_global = self.args.get("niter_fix_global", 0)
        self.lambda_feat = self.args.get("lambda_feat", 10)
        self.lambda_flow = self.args.get("lambda_flow", 1)
        loss = self.args.get("loss", LOSS)
        if loss not in ("ctc", "transformer"):
            self.loss_fn = getattr(torch.nn.functional, loss)

        #self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        #self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        # define loss functions
        #self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_flow_loss)
        
        
        # Names so we can breakout loss
        #self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_flow', 'D_real', 'D_fake')

    @staticmethod
    def add_to_argparse(parser):
        #parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')        
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=BETA_1, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=LR, help='initial learning rate for adam')

        #Todo: move to Subclass.

        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--no_flow_loss', action='store_true', help='if specified, do *not* use PWC-flow feature matching loss')
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        # for discriminators        
        parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')

        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_flow', type=float, default=1, help='weight for flow matching loss')
        return parser

    def configure_optimizers(self):
        #Todo: handle lr updation if that's not handled yet.
        # initialize optimizers
        # optimizer G
        if self.niter_fix_global > 0:                
            import sys
            if sys.version_info >= (3,0):
                finetune_list = set()
            else:
                from sets import Set
                finetune_list = Set()

            params_dict = dict(self.model.netG.named_parameters())
            params = []
            for key, value in params_dict.items():       
                if key.startswith('model' + str(self.args.get("n_local_enhancers", 1))):                    
                    params += [value]
                    finetune_list.add(key.split('.')[0])  
            print('------------- Only training the local enhancer network (for %d epochs) ------------' % self.args.niter_fix_global)
            print('The layers that are finetuned are ', sorted(finetune_list))                         
        else:
            params = list(self.model.netG.parameters())
        # if self.gen_features:              
        #     params += list(self.netE.parameters())         
        self.optimizer_G = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))                            

        # optimizer D                        
        params = list(self.netD.parameters())    
        self.optimizer_D = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))
        return [self.optimizer_G, self.optimizer_D], []
        #optimizer = self.optimizer_class(self.parameters(), lr=self.lr)

        # Todo: integrate lr scheduler.
        # if self.one_cycle_max_lr is None:
        #     return optimizer
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        # )
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)