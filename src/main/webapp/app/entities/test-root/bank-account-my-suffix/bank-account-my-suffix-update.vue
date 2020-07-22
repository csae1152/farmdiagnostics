<template>
    <div class="row justify-content-center">
        <div class="col-8">
            <form name="editForm" role="form" novalidate v-on:submit.prevent="save()" >
                <h2 id="jhipsterApp.testRootBankAccount.home.createOrEditLabel" v-text="$t('jhipsterApp.testRootBankAccount.home.createOrEditLabel')">Create or edit a BankAccountMySuffix</h2>
                <div>
                    <div class="form-group" v-if="bankAccount.id">
                        <label for="id" v-text="$t('global.field.id')">ID</label>
                        <input type="text" class="form-control" id="id" name="id"
                               v-model="bankAccount.id" readonly />
                    </div>
                     <div class="form-group">
                        <label class="form-control-label">Diseas</label>
                        <input type="text" class="form-control" name="name" id="bank-account-my-suffix-name"
                            :class="{'valid': !$v.bankAccount.description.$invalid, 'invalid': $v.bankAccount.description.$invalid }" v-model="$v.bankAccount.description.$model"  required/>
                        <div v-if="$v.bankAccount.description.$anyDirty && $v.bankAccount.description.$invalid">
                            <small class="form-text text-danger" v-if="!$v.bankAccount.description.required" v-text="$t('entity.validation.required')">
                                This field is required.
                            </small>
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-control-label" v-text="$t('jhipsterApp.testRootBankAccount.name')" for="bank-account-my-suffix-name">Name</label>
                        <input type="text" class="form-control" name="name" id="bank-account-my-suffix-name"
                            :class="{'valid': !$v.bankAccount.name.$invalid, 'invalid': $v.bankAccount.name.$invalid }" v-model="$v.bankAccount.name.$model"  required/>
                        <div v-if="$v.bankAccount.name.$anyDirty && $v.bankAccount.name.$invalid">
                            <small class="form-text text-danger" v-if="!$v.bankAccount.name.required" v-text="$t('entity.validation.required')">
                                This field is required.
                            </small>
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-control-label" v-text="$t('jhipsterApp.testRootBankAccount.balance')" for="bank-account-my-suffix-balance">Balance</label>
                        <input type="number" class="form-control" name="balance" id="bank-account-my-suffix-balance"
                            :class="{'valid': !$v.bankAccount.balance.$invalid, 'invalid': $v.bankAccount.balance.$invalid }" v-model.number="$v.bankAccount.balance.$model"  required/>
                        <div v-if="$v.bankAccount.balance.$anyDirty && $v.bankAccount.balance.$invalid">
                            <small class="form-text text-danger" v-if="!$v.bankAccount.balance.required" v-text="$t('entity.validation.required')">
                                This field is required.
                            </small>
                            <small class="form-text text-danger" v-if="!$v.bankAccount.balance.number" v-text="$t('entity.validation.number')">
                                This field should be a number.
                            </small>
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-control-label" v-text="$t('jhipsterApp.testRootBankAccount.openingDay')" for="bank-account-my-suffix-openingDay">Opening Day</label>
                        <div class="input-group">
                            <input id="bank-account-my-suffix-openingDay" type="date" class="form-control" name="openingDay"  :class="{'valid': !$v.bankAccount.openingDay.$invalid, 'invalid': $v.bankAccount.openingDay.$invalid }"
                            v-model="$v.bankAccount.openingDay.$model"  />
                        </div>
                    </div>                               
                    <div class="form-group">
                        <label class="form-control-label" v-text="$t('jhipsterApp.testRootBankAccount.attachment')" for="bank-account-my-suffix-attachment">Attachment</label>
                        <div>
                            <div v-if="bankAccount.attachment" class="form-text text-danger clearfix">
                                <a class="pull-left" v-on:click="openFile(bankAccount.attachmentContentType, bankAccount.attachment)" v-text="$t('entity.action.open')">open</a><br>
                                <span class="pull-left">{{bankAccount.attachmentContentType}}, {{byteSize(bankAccount.attachment)}}</span>
                                <button type="button" v-on:click="bankAccount.attachment=null;bankAccount.attachmentContentType=null;"
                                        class="btn btn-secondary btn-xs pull-right">
                                    <font-awesome-icon icon="times"></font-awesome-icon>
                                </button>
                            </div>
                            <input type="file" ref="file_attachment" id="file_attachment" v-on:change="setFileData($event, bankAccount, 'attachment', false)" v-text="$t('entity.action.addblob')"/>
                        </div>
                        <input type="hidden" class="form-control" name="attachment" id="bank-account-my-suffix-attachment"
                            :class="{'valid': !$v.bankAccount.attachment.$invalid, 'invalid': $v.bankAccount.attachment.$invalid }" v-model="$v.bankAccount.attachment.$model" />
                        <input type="hidden" class="form-control" name="attachmentContentType" id="bank-account-my-suffix-attachmentContentType"
                            v-model="bankAccount.attachmentContentType" />
                    </div>             
                 
                </div>
                <div>
                    <button type="button" id="cancel-save" class="btn btn-secondary" v-on:click="previousState()">
                        <font-awesome-icon icon="ban"></font-awesome-icon>&nbsp;<span v-text="$t('entity.action.cancel')">Cancel</span>
                    </button>
                    <button type="submit" id="save-entity" :disabled="$v.bankAccount.$invalid || isSaving" class="btn btn-primary">
                        <font-awesome-icon icon="save"></font-awesome-icon>&nbsp;<span v-text="$t('entity.action.save')">Save</span>
                    </button>
                </div>
            </form>
        </div>
    </div>
</template>
<script lang="ts" src="./bank-account-my-suffix-update.component.ts">
</script>
