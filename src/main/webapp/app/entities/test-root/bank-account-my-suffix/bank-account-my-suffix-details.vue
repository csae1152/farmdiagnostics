<template>
    <div class="row justify-content-center">
        <div class="col-8">
            <div v-if="bankAccount">
                <h2 class="jh-entity-heading"><span v-text="$t('jhipsterApp.testRootBankAccount.detail.title')">BankAccountMySuffix</span> {{bankAccount.id}}</h2>
                <dl class="row jh-entity-details">
                    <dt>
                        <span v-text="$t('jhipsterApp.testRootBankAccount.name')">Name</span>
                    </dt>
                    <dd>
                        <span>{{bankAccount.name}}</span>
                    </dd>
                    <dt>
                        <span v-text="$t('jhipsterApp.testRootBankAccount.balance')">Balance</span>
                    </dt>
                    <dd>
                        <span>{{bankAccount.balance}}</span>
                    </dd>
                    <dt>
                        <span v-text="$t('jhipsterApp.testRootBankAccount.attachment')">Attachment</span>
                    </dt>
                    <dd>
                        <div v-if="bankAccount.attachment">
                            <a v-on:click="openFile(bankAccount.attachmentContentType, bankAccount.attachment)" v-text="$t('entity.action.open')">open</a>
                            {{bankAccount.attachmentContentType}}, {{byteSize(bankAccount.attachment)}}
                        </div>
                    </dd>                   
                </dl>
                <button type="submit"
                        v-on:click.prevent="previousState()"
                        class="btn btn-info">
                    <font-awesome-icon icon="arrow-left"></font-awesome-icon>&nbsp;<span v-text="$t('entity.action.back')"> Back</span>
                </button>
                <router-link v-if="bankAccount.id" :to="{name: 'BankAccountMySuffixEdit', params: {bankAccountId: bankAccount.id}}" tag="button" class="btn btn-primary">
                    <font-awesome-icon icon="pencil-alt"></font-awesome-icon>&nbsp;<span v-text="$t('entity.action.edit')"> Edit</span>
                </router-link>
            </div>
        </div>
    </div>
</template>

<script lang="ts" src="./bank-account-my-suffix-details.component.ts">
</script>
